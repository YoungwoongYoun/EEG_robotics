# =========================
# (sequential inference: one sequence per tick)
# =========================
import os

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from ament_index_python.packages import get_package_share_directory

import torch

from bci_interface.tcformer_module import TCFormer


class BCINode(Node):
    def __init__(self):
        super().__init__("bci_node")

        # ----------------------------
        # 1) Parameters (declare)
        # ----------------------------
        self.declare_parameter("cmd_topic", "/cmd_vel_bci")
        self.declare_parameter("dwell_sec", 4.0)
        self.declare_parameter("loop", True)
        self.declare_parameter("exit_on_done", False)

        self.declare_parameter("mode", "infer")  # replay_labels | infer
        self.declare_parameter("pt_file", "selected_timeseries_with_labels.pt")
        self.declare_parameter("checkpoint", "")  # .pt or .pth

        self.declare_parameter("class_names", ["LEFT", "RIGHT", "FEET", "TONGUE"])
        self.declare_parameter("v_list", [0.0, 0.0, 0.3, 0.0])
        self.declare_parameter("w_list", [0.3927, -0.3927, 0.0, 0.0])

        # ----------------------------
        # 2) Parameters (read/finalize)
        # ----------------------------
        self.cmd_topic = self.get_parameter("cmd_topic").value
        self.dwell_sec = float(self.get_parameter("dwell_sec").value)
        self.loop = bool(self.get_parameter("loop").value)
        self.exit_on_done = bool(self.get_parameter("exit_on_done").value)

        self.mode = self.get_parameter("mode").value
        self.pt_file = self.get_parameter("pt_file").value
        self.checkpoint = self.get_parameter("checkpoint").value

        self.class_names = list(self.get_parameter("class_names").value)
        self.v_list = [float(x) for x in self.get_parameter("v_list").value]
        self.w_list = [float(x) for x in self.get_parameter("w_list").value]

        # ----------------------------
        # 3) Publisher
        # ----------------------------
        self.pub = self.create_publisher(Twist, self.cmd_topic, 10)

        # ----------------------------
        # 4) Load data + (optional) load model
        # ----------------------------
        self.timeseries, self.labels = self._load_pt()
        self.B = int(self.timeseries.shape[0]) if self.timeseries is not None else 0

        self.model = None
        if self.mode == "infer":
            if not self.checkpoint:
                raise RuntimeError("mode=infer but checkpoint is empty.")
            self.model = self._load_model(self.checkpoint)

        self.idx = 0
        self.done = False

        self.get_logger().info(
            f"mode={self.mode}, dwell={self.dwell_sec}s, loop={self.loop}, exit_on_done={self.exit_on_done}, "
            f"B={self.B}, pub={self.cmd_topic}"
        )

        # ----------------------------
        # 5) Timer
        # ----------------------------
        self.timer = self.create_timer(self.dwell_sec, self._tick)

    # ---------- file/path helpers ----------
    def _share_config_path(self, filename: str) -> str:
        share = get_package_share_directory("bci_interface")
        return os.path.join(share, "config", filename)

    def _load_pt(self):
        path = self._share_config_path(self.pt_file)
        obj = torch.load(path, map_location="cpu")
        x = obj["timeseries"]  # expected: [B,1,22,T]
        y = obj["labels"]      # expected: [B]
        return x, y

    # ---------- model helpers ----------
    def _load_model(self, checkpoint_filename: str):
        ckpt_path = self._share_config_path(checkpoint_filename)
        model = TCFormer(n_channels=22, n_classes=4)

        sd = torch.load(ckpt_path, map_location="cpu")

        # Support common wrapped checkpoint formats:
        # 1) pure state_dict (OrderedDict of parameter tensors)
        # 2) {"state_dict": ...}
        # 3) {"model": ...}
        if isinstance(sd, dict):
            if "state_dict" in sd and isinstance(sd["state_dict"], dict):
                sd = sd["state_dict"]
            elif "model" in sd and isinstance(sd["model"], dict):
                sd = sd["model"]

        model.load_state_dict(sd)
        model.eval()
        return model

    # ---------- publishing ----------
    def _publish_twist(self, v: float, w: float, tag: str = ""):
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.pub.publish(msg)
        if tag:
            self.get_logger().info(f"{tag} -> v={v:.3f}, w={w:.3f}")

    def _publish_for_class_id(self, class_id: int):
        if class_id < 0 or class_id >= len(self.class_names):
            name, v, w = "TONGUE", 0.0, 0.0
        else:
            name = self.class_names[class_id]
            v = self.v_list[class_id] if class_id < len(self.v_list) else 0.0
            w = self.w_list[class_id] if class_id < len(self.w_list) else 0.0
        self._publish_twist(v, w, tag=f"class_id={class_id}({name})")

    # ---------- done/exit policy ----------
    def _finish(self):
        self.done = True

        # Safety stop once
        self._publish_twist(0.0, 0.0, tag="DONE(stop)")

        # Stop timer
        try:
            self.timer.cancel()
        except Exception:
            pass

        # Exit process if requested
        if self.exit_on_done:
            self.get_logger().info("exit_on_done=true -> shutting down.")
            rclpy.shutdown()

    # ---------- main tick ----------
    def _tick(self):
        if self.done:
            return

        if self.B <= 0:
            self._finish()
            return

        # End-of-sequence handling BEFORE doing any work
        if self.idx >= self.B:
            if self.loop:
                self.idx = 0
            else:
                self._finish()
                return

        # 1) pick one sequence (trial/window)
        x_i = self.timeseries[self.idx:self.idx + 1]  # [1,1,22,T]

        # 2) decide class_id
        if self.mode == "replay_labels":
            cid = int(self.labels[self.idx].item())
        elif self.mode == "infer":
            with torch.no_grad():
                logits = self.model(x_i)             # [1,4]
                cid = int(logits.argmax(dim=1).item())
        else:
            raise RuntimeError(f"Unknown mode: {self.mode}")

        self.get_logger().info(f"idx={self.idx}/{self.B} -> cid={cid}")

        # 3) publish Twist based on class_id
        self._publish_for_class_id(cid)

        # 4) advance index
        self.idx += 1


def main():
    rclpy.init()
    node = BCINode()

    # Debug logs
    node.get_logger().info(f"class_names={node.class_names}")
    node.get_logger().info(f"v_list={node.v_list}, w_list={node.w_list}")
    try:
        uniq = sorted(set(int(i) for i in node.labels.tolist()))
    except Exception:
        uniq = []
    node.get_logger().info(f"labels unique={uniq}")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # If shutdown already called inside node, rclpy.ok() will be False
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
