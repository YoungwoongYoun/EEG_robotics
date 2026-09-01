"""Build the journal Supplementary Information DOCX from manuscript assets."""

from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Callable

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS = PROJECT_ROOT / "manuscript_assets"


def _read_csv(relative: str) -> list[dict[str, str]]:
    path = ASSETS / relative
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing manuscript asset: {path}. Run scripts/build_manuscript_assets.py first."
        )
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty manuscript asset: {path}")
    return rows


def _float(value: str, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _p(value: str) -> str:
    number = float(value)
    return f"{number:.3g}" if number < 0.001 else f"{number:.4f}"


def _table(
    title: str,
    note: str,
    headers: list[str],
    rows: list[dict[str, str]],
    cells: list[Callable[[dict[str, str]], Any]],
    *,
    compact: bool = False,
) -> str:
    if len(headers) != len(cells):
        raise ValueError(f"Header/cell mismatch for {title}")
    body = []
    for row in rows:
        body.append(
            "<tr>" + "".join(f"<td>{html.escape(str(cell(row)))}</td>" for cell in cells) + "</tr>"
        )
    classes = "data compact" if compact else "data"
    return f"""
<h2>{html.escape(title)}</h2>
<table class="{classes}">
  <thead><tr>{''.join(f'<th>{html.escape(header)}</th>' for header in headers)}</tr></thead>
  <tbody>{''.join(body)}</tbody>
</table>
<p class="table-note">{html.escape(note)}</p>
"""


def _figure(number: str, filename: str, caption: str) -> str:
    path = ASSETS / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing supplementary figure: {path}")
    with Image.open(path) as source:
        image = source.copy()
    image.thumbnail((560, 620), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    uri = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"""
<br clear="all"><p class="hard-break">&nbsp;</p><div class="figure-page">
  <p class="figure-heading"><strong>Supplementary Figure {html.escape(number)}</strong></p>
  <p class="figure"><img width="{image.width}" height="{image.height}" src="{html.escape(uri)}" alt="Supplementary Figure {html.escape(number)}"></p>
  <p class="caption"><strong>Supplementary Figure {html.escape(number)}.</strong> {html.escape(caption)}</p>
</div><br clear="all">
"""


def build_html() -> str:
    methods = _read_csv("tables/table_02_restoration_methods.csv")
    primary = _read_csv("statistics/classifier_primary_paired.csv")
    subjects = _read_csv("supplementary/table_s02_subject_accuracy.csv")
    signal = _read_csv("statistics/signal_endpoint_paired.csv")
    frozen = _read_csv("supplementary/table_s01_frozen_oracle.csv")
    ablation = _read_csv("tables/table_06_ae_ablation.csv")
    ablation_stats = _read_csv("statistics/ae_ablation_paired.csv")
    latency = _read_csv("tables/table_07_processing_cost.csv")

    method_labels = {
        "direct_mi9": "Direct MI-9",
        "zero_padded_mi9": "Zero-padded MI-9",
        "spherical_spline": "Spherical spline",
        "autoencoder": "Autoencoder",
        "autoencoder_bandpower": "AE + bandpower",
        "autoencoder_spatial": "AE + spatial",
        "autoencoder_eeg_aware": "EEG-aware AE",
        "ddpm_standard": "Standard DDPM",
        "wgan_gp": "Conditional WGAN-GP",
        "true22": "True 22-channel",
    }
    endpoint_labels = {
        "bandpower_mse": "Class-conditional mu/beta power MSE",
        "covariance_distance": "Covariance AIRM",
        "csp_feature_mse": "CSP feature MSE",
    }

    parts = ["""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Supplementary Information</title>
<style>
@page { size: A4; margin: 18mm 17mm 18mm 17mm; }
body { font-family: Arial, Helvetica, sans-serif; font-size: 9.5pt; line-height: 1.25; color: #111; }
h1 { font-size: 20pt; text-align: center; margin: 30mm 0 8mm; }
h2 { font-size: 12pt; margin: 7mm 0 2.5mm; padding: 0; text-indent: 0; clear: both; page-break-after: avoid; }
h3 { font-size: 10.5pt; margin: 5mm 0 2mm; page-break-after: avoid; }
p { margin: 2mm 0; }
.subtitle { text-align: center; font-size: 12pt; margin-bottom: 14mm; }
.scope { border: 1px solid #777; background: #f5f5f5; padding: 3mm; }
table.data { border-collapse: collapse; width: 100%; margin: 2mm 0 1.5mm; page-break-inside: avoid; }
table.data th, table.data td { border: 0.5pt solid #777; padding: 1.4mm; vertical-align: top; }
table.data th { background: #e9eef3; font-weight: bold; text-align: center; }
table.compact { font-size: 7.7pt; }
.table-note { font-size: 8.2pt; margin-bottom: 5mm; }
.hard-break { page-break-before: always; height: 1pt; margin: 0; padding: 0; font-size: 1pt; }
.figure-page { page-break-before: avoid; clear: both; width: 100%; }
.figure-heading { font-size: 12pt; margin: 7mm 0 2.5mm; padding: 0; text-indent: 0; clear: both; }
.figure { text-align: center; margin: 2mm 0; }
.figure img { display: block; float: none; margin: 0 auto; max-width: 158mm; max-height: 178mm; }
.caption { font-size: 9pt; text-align: justify; }
.page-break { page-break-before: always; }
ul { margin-top: 2mm; }
</style></head><body>
<h1>Supplementary Information</h1>
<p class="subtitle"><strong>Comparative Evaluation of EEG Channel Restoration for Low-Channel Motor-Imagery Decoding</strong></p>
<p class="scope"><strong>Scope.</strong> This document contains detailed reproducibility settings, subject-level results, complete inferential statistics, diagnostic frozen-oracle results, objective ablations, and all-subject signal analyses. The main manuscript retains only the study design, primary matched-classification results, the central signal-task comparison, and the formal processing-cost summary.</p>
<h2>Supplementary Methods</h2>
<p>All restoration models and matched classifiers were developed with pooled Session-1 data. Checkpoint selection used Session-1 validation data only, and Session 2 was held out for final evaluation. The same nine-subject cohort appears across sessions; the protocol therefore measures inter-session rather than unseen-subject generalization. Classifier statistics use subject as the inferential unit after averaging five classifier seeds within each subject.</p>
"""]

    parts.append(_table(
        "Supplementary Table S1a. Restoration families and objectives",
        "WGAN-GP inference excludes the critic. Inference parameter counts describe the restoration module only.",
        ["Method", "Family", "Objective", "Inference parameters"],
        methods,
        [
            lambda r: r["label"], lambda r: r["family"], lambda r: r["objective"],
            lambda r: f"{int(r['restoration_inference_parameters']):,}",
        ],
    ))
    parts.append(_table(
        "Supplementary Table S1b. Training and inference settings",
        "The interpolation method has no learned checkpoint. DDPM inference uses the fixed DDIM-100 sampler.",
        ["Method", "Optimizer / LR", "Batch", "Max / selected epoch", "Stop reason", "Sampler / steps"],
        methods,
        [
            lambda r: r["label"], lambda r: f"{r['optimizer']} / {r['learning_rate']}",
            lambda r: r["batch_size"],
            lambda r: f"{r['max_epochs']} / {r['selected_epoch']}", lambda r: r["stop_reason"],
            lambda r: f"{r['sampler']} / {r['sampling_steps']}",
        ],
    ))

    parts.append('<div class="page-break"></div><h2>Supplementary Results</h2>')
    parts.append(_table(
        "Supplementary Table S2. Prespecified paired classifier comparisons",
        "Differences are reference minus comparison in percentage points. Confidence intervals use 10,000 paired subject bootstrap resamples. Two-sided Wilcoxon p-values are Holm-corrected across the six prespecified comparisons. Positive rank-biserial values favor the reference.",
        ["Comparison", "Difference (pp)", "95% CI", "Wins", "Rank-biserial", "Cohen's dz", "Raw p", "Holm p"],
        primary,
        [
            lambda r: f"{method_labels[r['reference']]} vs {method_labels[r['compared']]}",
            lambda r: _float(r["mean_difference_accuracy_pp"], 2),
            lambda r: f"[{_float(r['bootstrap_ci95_low_pp'], 2)}, {_float(r['bootstrap_ci95_high_pp'], 2)}]",
            lambda r: f"{r['reference_wins']}/{r['n_subjects']}",
            lambda r: _float(r["rank_biserial_reference_better"], 3),
            lambda r: _float(r["cohen_dz"], 3), lambda r: _p(r["wilcoxon_raw_p"]),
            lambda r: _p(r["holm_adjusted_p"]),
        ],
        compact=True,
    ))

    subject_headers = ["Subject"] + [method_labels[name] + " (%)" for name in (
        "direct_mi9", "zero_padded_mi9", "spherical_spline", "autoencoder",
        "autoencoder_eeg_aware", "ddpm_standard", "wgan_gp", "true22",
    )]
    subject_cells: list[Callable[[dict[str, str]], Any]] = [lambda r: r["subject_id"]]
    for method in (
        "direct_mi9", "zero_padded_mi9", "spherical_spline", "autoencoder",
        "autoencoder_eeg_aware", "ddpm_standard", "wgan_gp", "true22",
    ):
        subject_cells.append(lambda r, method=method: _float(r[f"{method}_accuracy_percent"], 2))
    parts.append(_table(
        "Supplementary Table S3. Subject-level matched accuracy",
        "Each value is the subject's mean Session-2 accuracy across five independently trained TCFormer seeds.",
        subject_headers, subjects, subject_cells, compact=True,
    ))

    parts.append(_table(
        "Supplementary Table S4. Task-relevant signal endpoint statistics",
        "The EEG-aware autoencoder is the reference and lower endpoint error is better. Confidence intervals use 10,000 paired subject bootstrap resamples. Holm correction is applied separately within each endpoint family.",
        ["Endpoint", "Comparison", "Difference", "95% CI", "Reference wins", "Rank-biserial", "Raw p", "Holm p"],
        signal,
        [
            lambda r: endpoint_labels[r["endpoint"]], lambda r: method_labels[r["comparison"]],
            lambda r: _float(r["mean_difference_comparison_minus_reference"], 5),
            lambda r: f"[{_float(r['bootstrap_ci_low'], 5)}, {_float(r['bootstrap_ci_high'], 5)}]",
            lambda r: f"{r['reference_wins']}/{r['n_subjects']}",
            lambda r: _float(r["rank_biserial_reference_better"], 3),
            lambda r: _p(r["wilcoxon_raw_p"]), lambda r: _p(r["holm_adjusted_p"]),
        ],
        compact=True,
    ))

    parts.append(_table(
        "Supplementary Table S5. Frozen-oracle diagnostic",
        "The True-22 TCFormer is held fixed and receives each restored input. This is a representation-shift diagnostic, not the primary classification endpoint.",
        ["Method", "Frozen accuracy (%)", "Oracle agreement (%)", "Probability L1"],
        frozen,
        [
            lambda r: r["label"],
            lambda r: f"{_float(r['frozen_accuracy_mean_percent'], 2)} ± {_float(r['frozen_accuracy_subject_sd_percent'], 2)}",
            lambda r: f"{_float(r['oracle_agreement_mean_percent'], 2)} ± {_float(r['oracle_agreement_subject_sd_percent'], 2)}",
            lambda r: f"{_float(r['probability_l1_mean'], 4)} ± {_float(r['probability_l1_subject_sd'], 4)}",
        ],
    ))

    parts.append(_table(
        "Supplementary Table S6. EEG-aware autoencoder objective ablation",
        "Values are mean ± subject SD after averaging five classifier seeds within subject. All variants use the same architecture and capacity.",
        ["Variant", "Loss weights", "Accuracy (%)", "Macro-F1 (%)", "Kappa"],
        ablation,
        [
            lambda r: r["label"], lambda r: r["loss_weights"],
            lambda r: f"{_float(r['accuracy_mean_percent'], 2)} ± {_float(r['accuracy_subject_sd_percent'], 2)}",
            lambda r: f"{_float(r['macro_f1_mean_percent'], 2)} ± {_float(r['macro_f1_subject_sd_percent'], 2)}",
            lambda r: f"{_float(r['cohen_kappa_mean'], 3)} ± {_float(r['cohen_kappa_subject_sd'], 3)}",
        ],
    ))
    parts.append(_table(
        "Supplementary Table S7. Paired objective-ablation statistics",
        "The combined EEG-aware objective is the reference. Wilcoxon p-values are Holm-corrected across the three objective comparisons.",
        ["Comparison", "Difference (pp)", "95% CI", "Wins", "Rank-biserial", "Cohen's dz", "Holm p"],
        ablation_stats,
        [
            lambda r: f"{method_labels[r['reference']]} vs {method_labels[r['compared']]}",
            lambda r: _float(r["mean_difference_accuracy_pp"], 2),
            lambda r: f"[{_float(r['bootstrap_ci95_low_pp'], 2)}, {_float(r['bootstrap_ci95_high_pp'], 2)}]",
            lambda r: f"{r['reference_wins']}/{r['n_subjects']}",
            lambda r: _float(r["rank_biserial_reference_better"], 3),
            lambda r: _float(r["cohen_dz"], 3), lambda r: _p(r["holm_adjusted_p"]),
        ],
        compact=True,
    ))

    parts.append(_table(
        "Supplementary Table S8. Detailed processing-cost benchmark",
        "Batch size 1 on NVIDIA RTX A6000. Processing begins with a CPU-resident trial and ends with the predicted class returned to CPU. It excludes model loading, file I/O, EEG acquisition, ROS2 transport, safety filtering, and robot actuation.",
        ["Method", "Restoration median / p95 (ms)", "Processing median / p95 (ms)", "Inference parameters", "Peak allocated GPU (MB)", "Steps"],
        latency,
        [
            lambda r: r["label"],
            lambda r: f"{_float(r['restoration_median_ms'])} / {_float(r['restoration_p95_ms'])}",
            lambda r: f"{_float(r['processing_median_ms'])} / {_float(r['processing_p95_ms'])}",
            lambda r: f"{int(r['total_inference_parameters']):,}",
            lambda r: _float(r["peak_allocated_gpu_mb"], 1), lambda r: r["sampling_steps"],
        ],
        compact=True,
    ))

    parts.extend([
        _figure(
            "S1", "figures/figure_05_ae_ablation.png",
            "EEG-aware autoencoder objective ablation. Each point is one subject's mean across five TCFormer seeds; error bars show 95% bootstrap confidence intervals across nine subjects. The combined objective achieved the highest mean, but none of the three paired contrasts remained significant after Holm correction.",
        ),
        _figure(
            "S2", "supplementary/figures/figure_s01_true22_class_bandpower_topography.png",
            "True 22-channel class-conditional mu- and beta-band topographies averaged over all nine subjects. No favorable subject was selected.",
        ),
        _figure(
            "S3", "supplementary/figures/figure_s02_restoration_bandpower_error_topography.png",
            "Class-conditional bandpower-error topographies across the retained restoration methods, averaged over all nine subjects and four motor-imagery classes.",
        ),
        _figure(
            "S4", "supplementary/figures/figure_s03_restoration_time_frequency_error.png",
            "Grand-average time-frequency restoration error across all subjects and classes. The same held-out Session-2 trials are used for every method.",
        ),
        _figure(
            "S5", "supplementary/figures/figure_s04_signal_metric_summary.png",
            "Summary of class-conditional spectral, covariance, and CSP feature-preservation endpoints. Lower error or distance is better; these signal-space rankings need not match downstream classification rankings.",
        ),
    ])
    parts.append("</body></html>")
    return "".join(parts)


def build_docx(output: Path) -> Path:
    if shutil.which("libreoffice") is None:
        raise RuntimeError("LibreOffice is required to build Supplementary Information DOCX")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="supplementary-docx-") as temporary:
        html_path = Path(temporary) / "supplementary_information_source.html"
        html_path.write_text(build_html(), encoding="utf-8")
        profile = Path(temporary) / "profile"
        converted = Path(temporary) / "supplementary_information_source.docx"
        command = [
            "libreoffice", "--headless", f"-env:UserInstallation={profile.as_uri()}",
            "--convert-to", "docx:Office Open XML Text", "--outdir", temporary,
            str(html_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        if result.returncode != 0 or not converted.is_file():
            raise RuntimeError(
                "LibreOffice conversion failed:\n" + result.stdout + "\n" + result.stderr
            )
        os.replace(converted, output)
    _set_a4_page_size(output)
    return output


def _set_a4_page_size(path: Path) -> None:
    """Match the A4 page size used by the submitted manuscript."""

    with tempfile.NamedTemporaryFile(
        suffix=".docx", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    replaced = 0
    try:
        with zipfile.ZipFile(path, "r") as source, zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED
        ) as destination:
            for item in source.infolist():
                data = source.read(item.filename)
                if item.filename == "word/document.xml":
                    data, replaced = re.subn(
                        rb'<w:pgSz\b[^>]*/>',
                        b'<w:pgSz w:w="11906" w:h="16838"/>',
                        data,
                        count=1,
                    )
                destination.writestr(item, data)
        if replaced != 1:
            raise RuntimeError("Could not set the Supplementary Information page size to A4")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "supplementary information.docx",
    )
    args = parser.parse_args()
    output = build_docx(args.output)
    print(f"Supplementary Information complete: {output}")


if __name__ == "__main__":
    main()
