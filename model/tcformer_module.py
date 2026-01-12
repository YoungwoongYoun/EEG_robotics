# tcformer_module.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kexrnel_size, dilation=1, groups=1, bias=True):
        self._pad = (kernel_size - 1) * dilation
        super().__init__(in_channels, out_channels, kernel_size,
                         padding=self._pad, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        y = super().forward(x)
        if self._pad > 0:
            return y[..., :-self._pad]
        return y


class Conv1dWithConstraint(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, groups=1, max_norm=None, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, groups=groups, bias=bias)
        self.max_norm = max_norm

    def forward(self, x):
        if self.max_norm is not None:
            with torch.no_grad():
                w = self.weight.view(self.out_channels, -1)
                norms = w.norm(dim=1, keepdim=True)
                desired = torch.clamp(norms, max=self.max_norm)
                self.weight.data *= (desired / (1e-8 + norms)).view(self.out_channels, 1, 1)
        return super().forward(x)


class TCNBlock(nn.Module):
    def __init__(self, kernel_length=4, n_filters=32, dilation=1, n_groups=1, dropout=0.3):
        super().__init__()
        self.conv1 = CausalConv1d(n_filters, n_filters, kernel_size=kernel_length,
                                  dilation=dilation, groups=n_groups, bias=True)
        self.bn1 = nn.BatchNorm1d(n_filters)
        self.conv2 = CausalConv1d(n_filters, n_filters, kernel_size=kernel_length,
                                  dilation=dilation, groups=n_groups, bias=True)
        self.bn2 = nn.BatchNorm1d(n_filters)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.act = nn.ELU()

        nn.init.constant_(self.conv1.bias, 0.0)
        nn.init.constant_(self.conv2.bias, 0.0)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.act(y)
        y = self.drop1(y)

        y = self.conv2(y)
        y = self.bn2(y)
        y = self.act(y)
        y = self.drop2(y)

        return self.act(x + y)


class TCN(nn.Module):
    def __init__(self, depth=2, kernel_length=4, n_filters=32, n_groups=1, dropout=0.3):
        super().__init__()
        blocks = []
        for i in range(depth):
            dilation = 2 ** i
            blocks.append(TCNBlock(kernel_length, n_filters, dilation, n_groups, dropout))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class ClassificationHead(nn.Module):
    def __init__(self, d_features, n_groups, n_classes, kernel_size=1, max_norm=0.25):
        super().__init__()
        self.n_groups = n_groups
        self.n_classes = n_classes
        self.linear = Conv1dWithConstraint(
            in_channels=d_features,
            out_channels=n_classes * n_groups,
            kernel_size=kernel_size,
            groups=n_groups,
            max_norm=max_norm,
            bias=True,
        )

    def forward(self, x):
        x = self.linear(x).squeeze(-1)          # [B, n_classes * n_groups]
        x = x.view(x.size(0), self.n_groups, self.n_classes).mean(dim=1)
        return x                                # [B, n_classes]


class TCNHead(nn.Module):
    def __init__(self, d_features=64, n_groups=1, tcn_depth=2,
                 kernel_length=4, dropout_tcn=0.3, n_classes=4):
        super().__init__()
        self.tcn = TCN(tcn_depth, kernel_length, d_features, n_groups, dropout_tcn)
        self.classifier = ClassificationHead(
            d_features=d_features,
            n_groups=n_groups,
            n_classes=n_classes,
        )

    def forward(self, x):
        x = self.tcn(x)              # [B, d_features, T]
        x = x[:, :, -1:].contiguous()
        x = self.classifier(x)
        return x


class MultiKernelConvBlock(nn.Module):
    def __init__(
        self,
        n_channels,
        temp_kernel_lengths=(16, 32, 64),
        F1=16,
        D=2,
        pool_length_1=8,
        pool_length_2=7,
        dropout=0.3,
        d_group=16,
        use_group_attn=False,
    ):
        super().__init__()
        self.n_groups = len(temp_kernel_lengths)
        self.rearrange = None

        self.temporal_convs = nn.ModuleList([
            nn.Sequential(
                nn.ConstantPad2d(
                    (k // 2 - 1, k // 2, 0, 0) if k % 2 == 0 else (k // 2, k // 2, 0, 0),
                    0.0,
                ),
                nn.Conv2d(1, F1, (1, k), bias=False),
                nn.BatchNorm2d(F1),
            )
            for k in temp_kernel_lengths
        ])

        self.d_model = d_group * self.n_groups
        self.use_channel_reduction_1 = False

        F2 = F1 * self.n_groups * D
        self.channel_DW_conv = nn.Sequential(
            nn.Conv2d(F1 * self.n_groups, F2, (n_channels, 1),
                      bias=False, groups=F1 * self.n_groups),
            nn.BatchNorm2d(F2),
            nn.ELU(),
        )
        self.pool1 = nn.AvgPool2d((1, pool_length_1))
        self.drop1 = nn.Dropout(dropout)

        self.use_channel_reduction_2 = (self.d_model != F2)
        if self.use_channel_reduction_2:
            self.channel_reduction_2 = nn.Sequential(
                nn.Conv2d(F2, self.d_model, (1, 1),
                          bias=False, groups=self.n_groups),
                nn.BatchNorm2d(self.d_model),
            )

        self.temporal_conv_2 = nn.Sequential(
            nn.Conv2d(self.d_model, self.d_model, (1, 16),
                      padding="same", bias=False, groups=self.n_groups),
            nn.BatchNorm2d(self.d_model),
            nn.ELU(),
        )

        self.use_group_attn = False
        self.pool2 = nn.AvgPool2d((1, pool_length_2))
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 1, C, T]
        feats = [conv(x) for conv in self.temporal_convs]
        x = torch.cat(feats, dim=1)  # [B, F1*n_groups, C, T]

        x = self.channel_DW_conv(x)
        x = self.pool1(x)
        x = self.drop1(x)

        if self.use_channel_reduction_2:
            x = self.channel_reduction_2(x)

        x = self.temporal_conv_2(x)
        x = self.pool2(x)
        x = self.drop2(x)

        x = x.squeeze(2)  # [B, d_model, T_reduced]
        return x


def _build_rotary_cache(head_dim, seq_len, device):
    theta = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    seq_idx = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(seq_idx, theta)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos(), emb.sin()
    return cos, sin


def _rope(q, k, cos, sin):
    def _rotate(x):
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).reshape_as(x)

    q_out = q * cos + _rotate(q) * sin
    k_out = k * cos + _rotate(k) * sin
    return q_out, k_out


def _xavier_zero_bias(module):
    for m in module.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class _GQAttention(nn.Module):
    def __init__(self, d_model, num_q_heads, num_kv_heads, dropout=0.3):
        super().__init__()
        assert d_model % num_q_heads == 0
        assert num_q_heads % num_kv_heads == 0
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_q_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

        _xavier_zero_bias(self)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_q_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(x).view(B, T, self.num_kv_heads, 2, self.head_dim)
        k = kv[..., 0, :].transpose(1, 2)
        v = kv[..., 1, :].transpose(1, 2)

        repeat_factor = self.num_q_heads // self.num_kv_heads
        k = k.repeat_interleave(repeat_factor, dim=1)
        v = v.repeat_interleave(repeat_factor, dim=1)

        cos = cos[:T, :].unsqueeze(0).unsqueeze(0)
        sin = sin[:T, :].unsqueeze(0).unsqueeze(0)
        q, k = _rope(q, k, cos, sin)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


class _TransformerBlock(nn.Module):
    def __init__(self, d_model, q_heads, kv_heads,
                 mlp_ratio=2, dropout=0.4, drop_path_rate=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = _GQAttention(d_model, q_heads, kv_heads, dropout)
        self.drop_path = DropPath(drop_path_rate)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.GELU(),
            nn.Linear(mlp_ratio * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, cos, sin):
        x = x + self.drop_path(self.attn(self.norm1(x), cos, sin))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TCFormerModule(nn.Module):
    def __init__(
        self,
        n_channels,
        n_classes,
        F1=16,
        temp_kernel_lengths=(16, 32, 64),
        pool_length_1=8,
        pool_length_2=7,
        D=2,
        dropout_conv=0.3,
        d_group=16,
        tcn_depth=2,
        kernel_length_tcn=4,
        dropout_tcn=0.3,
        use_group_attn=False,
        q_heads=8,
        kv_heads=4,
        trans_depth=5,
        trans_dropout=0.4,
        drop_path_max=0.25,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.n_groups = len(temp_kernel_lengths)
        self.d_model = d_group * self.n_groups

        self.conv_block = MultiKernelConvBlock(
            n_channels=n_channels,
            temp_kernel_lengths=temp_kernel_lengths,
            F1=F1,
            D=D,
            pool_length_1=pool_length_1,
            pool_length_2=pool_length_2,
            dropout=dropout_conv,
            d_group=d_group,
            use_group_attn=use_group_attn,
        )

        self.mix = nn.Sequential(
            nn.Conv1d(self.d_model, self.d_model, kernel_size=1,
                      groups=1, bias=False),
            nn.BatchNorm1d(self.d_model),
            nn.SiLU(),
        )

        drop_rates = torch.linspace(0, 1, trans_depth) ** 2 * drop_path_max
        self.register_buffer("_cos", None, persistent=False)
        self.register_buffer("_sin", None, persistent=False)

        self.transformer = nn.ModuleList([
            _TransformerBlock(self.d_model, q_heads, kv_heads,
                              dropout=trans_dropout,
                              drop_path_rate=drop_rates[i].item())
            for i in range(trans_depth)
        ])

        self.reduce = nn.Sequential(
            nn.Conv1d(self.d_model, d_group, kernel_size=1,
                      groups=1, bias=False),
            nn.BatchNorm1d(d_group),
            nn.SiLU(),
        )

        self.tcn_head = TCNHead(
            d_features=d_group * (self.n_groups + 1),
            n_groups=(self.n_groups + 1),
            tcn_depth=tcn_depth,
            kernel_length=kernel_length_tcn,
            dropout_tcn=dropout_tcn,
            n_classes=n_classes,
        )

    def _rotary_cache(self, seq_len, device):
        head_dim = self.transformer[0].attn.head_dim
        if self._cos is None or self._cos.shape[0] < seq_len:
            cos, sin = _build_rotary_cache(head_dim, seq_len, device)
            self._cos, self._sin = cos.to(device), sin.to(device)
        return self._cos, self._sin

    def forward(self, x):
        conv_features = self.conv_block(x)         # [B, d_model, T']
        mixed = self.mix(conv_features)            # [B, d_model, T']
        tokens = mixed.permute(0, 2, 1)            # [B, T', d_model]

        T = tokens.size(1)
        cos, sin = self._rotary_cache(T, tokens.device)

        for blk in self.transformer:
            tokens = blk(tokens, cos, sin)

        tokens_t = tokens.permute(0, 2, 1)        # [B, d_model, T']
        tran_features = self.reduce(tokens_t)     # [B, d_group, T']

        features = torch.cat((conv_features, tran_features), dim=1)
        out = self.tcn_head(features)             # [B, n_classes]
        return out


class TCFormer(nn.Module):
    def __init__(self, n_channels, n_classes, **kwargs):
        super().__init__()
        self.net = TCFormerModule(n_channels=n_channels, n_classes=n_classes, **kwargs)

    def forward(self, x):
        x = x.squeeze(1)      # [B, 22, T]
        return self.net(x)


def summarize_model(model, input_shape, device):
    model = model.to(device)
    model.eval()
    x = torch.randn(*((1,) + input_shape), device=device)
    with torch.no_grad():
        out = model(x)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print(f"Input shape : {tuple(x.shape)}")
    print(f"Output shape: {tuple(out.shape)}")
    print(f"Trainable parameters: {n_params:,}")
