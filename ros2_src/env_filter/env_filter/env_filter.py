#!/usr/bin/env python3
import math
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class EnvFilter(Node):
    def __init__(self):
        super().__init__("env_filter")

        # -------------------------
        # Parameters
        # -------------------------
        self.declare_parameter("D_safe", 1.0)
        self.declare_parameter("D_stop", 0.35)
        self.declare_parameter("k", 0.3)

        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("cmd_in_topic", "/cmd_vel_bci")
        self.declare_parameter("cmd_out_topic", "/cmd_vel")

        self.declare_parameter("front_angle_min_deg", -30.0)
        self.declare_parameter("front_angle_max_deg", 30.0)

        self.declare_parameter("range_min_clip", 0.01)
        self.declare_parameter("range_max_clip", 30.0)

        self.D_safe = float(self.get_parameter("D_safe").value)
        self.D_stop = float(self.get_parameter("D_stop").value)
        self.k = float(self.get_parameter("k").value)

        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.cmd_in_topic = str(self.get_parameter("cmd_in_topic").value)
        self.cmd_out_topic = str(self.get_parameter("cmd_out_topic").value)

        self.front_min_deg = float(self.get_parameter("front_angle_min_deg").value)
        self.front_max_deg = float(self.get_parameter("front_angle_max_deg").value)

        self.rmin_clip = float(self.get_parameter("range_min_clip").value)
        self.rmax_clip = float(self.get_parameter("range_max_clip").value)

        # -------------------------
        # State
        # -------------------------
        self.last_cmd_bci: Twist = Twist()
        self.last_d_min: Optional[float] = None
        self.last_scan_stamp = None

        scan_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        cmd_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.sub_scan = self.create_subscription(
            LaserScan, self.scan_topic, self.cb_scan, scan_qos
        )
        self.sub_cmd = self.create_subscription(
            Twist, self.cmd_in_topic, self.cb_cmd, cmd_qos
        )

        self.pub_cmd = self.create_publisher(Twist, self.cmd_out_topic, 10)

        # Timer: 출력 주기를 고정
        self.timer = self.create_timer(0.05, self.tick)

        self.get_logger().info(
            f"env_filter started | scan={self.scan_topic}, cmd_in={self.cmd_in_topic}, cmd_out={self.cmd_out_topic}"
        )

    def cb_cmd(self, msg: Twist):
        self.last_cmd_bci = msg

    def cb_scan(self, msg: LaserScan):
        # 전방 각도 윈도우를 인덱스로 변환
        angle_min = msg.angle_min
        angle_inc = msg.angle_increment
        n = len(msg.ranges)

        # deg -> rad
        a0 = math.radians(self.front_min_deg)
        a1 = math.radians(self.front_max_deg)

        # scan frame에서 0 rad이 정면
        i0 = int(round((a0 - angle_min) / angle_inc))
        i1 = int(round((a1 - angle_min) / angle_inc))

        i0 = max(0, min(n - 1, i0))
        i1 = max(0, min(n - 1, i1))
        if i0 > i1:
            i0, i1 = i1, i0

        dmin = float("inf")
        for r in msg.ranges[i0 : i1 + 1]:
            if r is None:
                continue
            if math.isinf(r) or math.isnan(r):
                continue
            r = clamp(float(r), self.rmin_clip, self.rmax_clip)
            if r < dmin:
                dmin = r

        # 유효 데이터가 없으면 inf 유지
        self.last_d_min = dmin if dmin != float("inf") else None
        self.last_scan_stamp = msg.header.stamp

    def apply_rule(self, cmd: Twist, d_min: Optional[float]) -> Twist:
        out = Twist()
        # 기본: 입력 복사
        out.linear.x = cmd.linear.x
        out.angular.z = cmd.angular.z

        # scan이 없거나 유효값 없으면: 보수적으로 정지시키고 싶으면 0으로 변경 가능
        if d_min is None:
            out.linear.x = 0.0
            out.angular.z = 0.0
            return out

        if d_min > self.D_safe:
            return out
        elif d_min > self.D_stop:
            out.linear.x = self.k * cmd.linear.x
            return out
        else:
            out.linear.x = 0.0
            out.angular.z = 0.0
            return out

    def tick(self):
        out = self.apply_rule(self.last_cmd_bci, self.last_d_min)
        self.pub_cmd.publish(out)

        # 디버깅 로그(너무 잦으면 느려짐) → 1초에 1번 정도만 찍고 싶으면 카운터 추가 가능
        self.get_logger().debug(
            f"d_min={self.last_d_min} | in(v={self.last_cmd_bci.linear.x:.3f}, w={self.last_cmd_bci.angular.z:.3f}) "
            f"-> out(v={out.linear.x:.3f}, w={out.angular.z:.3f})"
        )


def main():
    rclpy.init()
    node = EnvFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
