#!/usr/bin/env python3
"""
Multi-gas field simulator for Gazebo/Sim.

Publishes synthetic ppm readings for multiple gas types at topics gas/<name>/ppm.
Each gas has independent sources and falloff; robot pose is obtained via TF.

Parameters
----------
- world_frame: TF world frame (default: 'odom')
- base_frame: robot base frame (default: 'base_link')
- rate_hz: publish rate (default: 10.0)
- gases: list of dicts, each with:
    name: string, e.g., 'methane', 'lpg', 'co', 'air_quality'
    falloff: float (default 2.0)
    min_dist: float meters (default 0.2)
    noise_stddev: float ppm (default 0.0)
    sources: list of {x, y, z, strength}
"""

import math
import random
from typing import Any, Dict, List

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from std_msgs.msg import Float32
import tf2_ros


class GasFieldSimMulti(Node):
    def __init__(self) -> None:
        super().__init__('gas_field_sim_multi')

        self.world_frame = (
            self.declare_parameter('world_frame', 'odom').get_parameter_value().string_value
        )
        self.base_frame = (
            self.declare_parameter('base_frame', 'base_link').get_parameter_value().string_value
        )
        self.rate_hz = (
            self.declare_parameter('rate_hz', 10.0).get_parameter_value().double_value
        )

        raw_gases = self.declare_parameter('gases', [
            {'name': 'methane', 'falloff': 2.0, 'min_dist': 0.2, 'noise_stddev': 0.0,
             'sources': [{'x': 2.0, 'y': 0.0, 'z': 0.0, 'strength': 5e5}]},
            {'name': 'lpg', 'falloff': 2.0, 'min_dist': 0.2, 'noise_stddev': 0.0,
             'sources': [{'x': -2.0, 'y': 1.0, 'z': 0.0, 'strength': 5e5}]},
            {'name': 'co', 'falloff': 2.0, 'min_dist': 0.2, 'noise_stddev': 0.0,
             'sources': [{'x': 0.0, 'y': 2.0, 'z': 0.0, 'strength': 5e5}]},
            {'name': 'air_quality', 'falloff': 2.0, 'min_dist': 0.2, 'noise_stddev': 0.0,
             'sources': [{'x': -1.5, 'y': -1.0, 'z': 0.0, 'strength': 5e5}]},
        ]).value
        self.gases: List[Dict[str, Any]] = self._parse_list(raw_gases)

        # Publishers per gas
        self.pubs: Dict[str, Any] = {}
        for g in self.gases:
            name = str(g.get('name', 'gas'))
            self.pubs[name] = self.create_publisher(Float32, f'gas/{name}/ppm', 10)

        # TF
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=5.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Timer
        self.timer = self.create_timer(max(0.01, 1.0 / max(0.1, self.rate_hz)), self._tick)

        self.get_logger().info(
            f"GasFieldSimMulti started world={self.world_frame} base={self.base_frame} gases={[g.get('name') for g in self.gases]}"
        )

    def _parse_list(self, raw):
        if isinstance(raw, list):
            if raw and isinstance(raw[0], str):
                try:
                    import yaml  # type: ignore
                    return yaml.safe_load('\n'.join(raw))
                except Exception:
                    return []
            return raw
        if isinstance(raw, str):
            try:
                import yaml  # type: ignore
                return yaml.safe_load(raw)
            except Exception:
                return []
        return []

    def _tick(self) -> None:
        try:
            tf = self.tf_buffer.lookup_transform(self.world_frame, self.base_frame, Time())
            x = tf.transform.translation.x
            y = tf.transform.translation.y
            z = tf.transform.translation.z
        except Exception as e:
            self.get_logger().debug(f'Waiting for TF {self.world_frame}->{self.base_frame}: {e}')
            return

        for g in self.gases:
            try:
                name = str(g.get('name', 'gas'))
                falloff = float(g.get('falloff', 2.0))
                min_dist = float(g.get('min_dist', 0.2))
                noise = float(g.get('noise_stddev', 0.0))
                sources = g.get('sources', [])

                ppm = 0.0
                for s in sources:
                    dx = x - float(s.get('x', 0.0))
                    dy = y - float(s.get('y', 0.0))
                    dz = z - float(s.get('z', 0.0))
                    d = max(min_dist, math.sqrt(dx * dx + dy * dy + dz * dz))
                    strength = float(s.get('strength', 0.0))
                    ppm += strength / (d ** falloff)
                if noise > 0.0:
                    ppm += random.gauss(0.0, noise)

                msg = Float32()
                msg.data = float(max(0.0, ppm))
                self.pubs[name].publish(msg)
            except Exception:
                continue


def main() -> None:
    rclpy.init()
    node = GasFieldSimMulti()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
