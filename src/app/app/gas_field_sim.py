#!/usr/bin/env python3
"""
Gas field simulator for JetAcker in Gazebo/Sim.

Publishes a synthetic gas concentration (ppm) at topic 'gas/ppm' based on
distance to configured gas sources in a world frame. Useful to prototype
gas-mapping and hazard behaviors without hardware sensors.

Parameters
----------
- world_frame: TF frame to treat as world (default: 'odom')
- base_frame: TF frame of robot base (default: 'base_link')
- sources: list of dicts [{x, y, z, strength}] for gas sources
- falloff: exponent of inverse distance falloff (default: 2.0)
- min_dist: clamp distance to avoid singularities (default: 0.2 m)
- rate_hz: publish rate in Hz (default: 10.0)
"""

import math
from typing import List, Dict, Any

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from std_msgs.msg import Float32
import tf2_ros


class GasFieldSim(Node):
    def __init__(self) -> None:
        super().__init__('gas_field_sim')

        # Parameters
        self.world_frame: str = (
            self.declare_parameter('world_frame', 'odom')
            .get_parameter_value()
            .string_value
        )
        self.base_frame: str = (
            self.declare_parameter('base_frame', 'base_link')
            .get_parameter_value()
            .string_value
        )

        # YAML list parameter parsing (accepts list of dicts or yaml string list)
        raw_sources = self.declare_parameter(
            'sources',
            [
                {'x': 2.0, 'y': 0.0, 'z': 0.0, 'strength': 5e5},
            ],
        ).value
        self.sources: List[Dict[str, Any]] = self._parse_sources(raw_sources)

        self.falloff: float = (
            self.declare_parameter('falloff', 2.0).get_parameter_value().double_value
        )
        self.min_dist: float = (
            self.declare_parameter('min_dist', 0.2).get_parameter_value().double_value
        )
        self.rate_hz: float = (
            self.declare_parameter('rate_hz', 10.0).get_parameter_value().double_value
        )

        # Publishers
        self.pub_ppm = self.create_publisher(Float32, 'gas/ppm', 10)

        # TF
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=5.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Timer
        self.timer = self.create_timer(max(0.01, 1.0 / self.rate_hz), self._tick)

        self.get_logger().info(
            f"GasFieldSim started. world_frame={self.world_frame} base_frame={self.base_frame} sources={self.sources}"
        )

    def _parse_sources(self, raw):
        # Already a list of dicts
        if isinstance(raw, list) and (not raw or isinstance(raw[0], dict)):
            return raw  # type: ignore
        # Provided as list of strings (YAML lines)
        if isinstance(raw, list) and raw and isinstance(raw[0], str):
            try:
                import yaml  # type: ignore

                return yaml.safe_load('\n'.join(raw))
            except Exception as e:
                self.get_logger().warn(f'Failed to parse sources YAML: {e}')
                return []
        # Single YAML string
        if isinstance(raw, str):
            try:
                import yaml  # type: ignore

                return yaml.safe_load(raw)
            except Exception as e:
                self.get_logger().warn(f'Failed to parse sources YAML: {e}')
                return []
        return []

    def _tick(self) -> None:
        try:
            tf = self.tf_buffer.lookup_transform(
                self.world_frame, self.base_frame, Time()
            )
            x = tf.transform.translation.x
            y = tf.transform.translation.y
            z = tf.transform.translation.z
        except Exception as e:
            self.get_logger().debug(
                f'Waiting for TF {self.world_frame}->{self.base_frame}: {e}'
            )
            return

        ppm = 0.0
        for s in self.sources:
            try:
                dx = x - float(s.get('x', 0.0))
                dy = y - float(s.get('y', 0.0))
                dz = z - float(s.get('z', 0.0))
                d = max(self.min_dist, math.sqrt(dx * dx + dy * dy + dz * dz))
                strength = float(s.get('strength', 0.0))
                ppm += strength / (d ** self.falloff)
            except Exception:
                continue

        msg = Float32()
        msg.data = float(ppm)
        self.pub_ppm.publish(msg)


def main() -> None:
    rclpy.init()
    node = GasFieldSim()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
