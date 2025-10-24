#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import random
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32


class GasSensorSim(Node):
    """Publish synthetic gas concentration readings for testing.

    Parameters:
    - topic (default: gas_sensor/concentration)
    - hz (default: 5.0)
    - min_ppm, max_ppm (default: 0, 400)
    """

    def __init__(self):
        super().__init__('gas_sensor_sim')
        self.declare_parameter('topic', 'gas_sensor/concentration')
        self.declare_parameter('hz', 5.0)
        self.declare_parameter('min_ppm', 0.0)
        self.declare_parameter('max_ppm', 400.0)

        topic = self.get_parameter('topic').value
        self.pub = self.create_publisher(Float32, topic, 10)
        hz = float(self.get_parameter('hz').value)
        self.min_ppm = float(self.get_parameter('min_ppm').value)
        self.max_ppm = float(self.get_parameter('max_ppm').value)
        period = 1.0 / max(0.1, hz)
        self.create_timer(period, self._tick)
        self.t0 = time.time()
        self.get_logger().info(f"gas_sensor_sim publishing on {topic} @ {hz:.1f}Hz")

    def _tick(self):
        # Smooth pseudo plume with noise
        t = time.time() - self.t0
        base = 0.5 * (1.0 + math.sin(t * 0.25))  # 0..1
        noise = 0.05 * random.random()
        val = self.min_ppm + (self.max_ppm - self.min_ppm) * min(1.0, max(0.0, base + noise))
        self.pub.publish(Float32(data=float(val)))


def main():
    rclpy.init()
    node = GasSensorSim()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
