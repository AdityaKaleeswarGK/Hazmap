#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time
from typing import Optional, Tuple, Dict, Any

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time

from std_msgs.msg import Float32
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import TransformStamped

from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ExtrapolationException, ConnectivityException


PPM_THRESHOLD_MIN = 20.0  # Edit this minimum ppm threshold as desired


class GasMapper(Node):
    """
    Fuse gas concentration readings with the current 2D map and robot pose to publish a gas map.

    - Subscribes:
        map_topic (nav_msgs/OccupancyGrid)
        gas_topic (std_msgs/Float32)  # concentration, e.g., ppm
    - Uses TF: map_frame -> base_frame
    - Publishes:
        publish_topic (nav_msgs/OccupancyGrid) as an intensity layer 0..100, -1 unknown

    Per-cell accumulation uses a running mean. Unknown cells remain -1.
    """

    def __init__(self) -> None:
        super().__init__('gas_mapper')

        # Parameters
        self.declare_parameter('map_topic', 'map')
        # Multi-gas configuration: list of {name, topic, max_concentration}
        self.declare_parameter('gases', [
            {'name': 'methane', 'topic': 'gas/methane/ppm', 'max_concentration': 10000.0},
            {'name': 'lpg', 'topic': 'gas/lpg/ppm', 'max_concentration': 10000.0},
            {'name': 'co', 'topic': 'gas/co/ppm', 'max_concentration': 10000.0},
            {'name': 'air_quality', 'topic': 'gas/air_quality/ppm', 'max_concentration': 1000.0},
        ])
        self.declare_parameter('publish_topic_prefix', 'gas_map')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('kernel_radius', 0)  # in cells
        self.declare_parameter('publish_hz', 2.0)
        self.declare_parameter('ppm_threshold', PPM_THRESHOLD_MIN)

        self.map_topic = self.get_parameter('map_topic').value
        self.publish_prefix = self.get_parameter('publish_topic_prefix').value
        self.map_frame = self.get_parameter('map_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.kernel_radius = int(self.get_parameter('kernel_radius').value)
        self.publish_hz = float(self.get_parameter('publish_hz').value)
        self.ppm_threshold = float(self.get_parameter('ppm_threshold').value)

        # Parse gases list param
        gases_param = self.get_parameter('gases').value
        self.gases = {}
        if isinstance(gases_param, list):
            for ent in gases_param:
                try:
                    name = str(ent['name'])
                    self.gases[name] = {
                        'topic': str(ent['topic']),
                        'max': float(ent.get('max_concentration', 1000.0)),
                    }
                except Exception:
                    continue

        # TF buffer/listener
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Internal state
        self.map_msg = None
        # Per-gas accumulators and state
        self.grid_mean = {k: None for k in self.gases.keys()}  # float32 (H,W)
        self.grid_count = {k: None for k in self.gases.keys()}  # uint32 (H,W)
        self.last_gas = {k: None for k in self.gases.keys()}
        self.last_gas_time = {k: 0.0 for k in self.gases.keys()}

        # IO
        self.map_sub = self.create_subscription(OccupancyGrid, self.map_topic, self._on_map, 1)
        # One subscriber and one publisher per gas
        self.gas_subs = {}
        self.pubs = {}
        for name, cfg in self.gases.items():
            topic = cfg['topic']
            self.gas_subs[name] = self.create_subscription(Float32, topic, self._make_gas_cb(name), 10)
            self.pubs[name] = self.create_publisher(OccupancyGrid, f"{self.publish_prefix}/{name}", 1)

        # Timers
        period = 1.0 / max(0.1, self.publish_hz)
        self.create_timer(period, self._publish_gas_maps)
        self.create_timer(0.2, self._accumulate_samples)

        self.get_logger().info(
            f"gas_mapper started; map: {self.map_topic}, gases: {list(self.gases.keys())} -> prefix: {self.publish_prefix}"
        )

    # --- Callbacks ---
    def _on_map(self, msg: OccupancyGrid) -> None:
        reinit = (
            self.map_msg is None
            or self.map_msg.info.width != msg.info.width
            or self.map_msg.info.height != msg.info.height
            or self.map_msg.info.resolution != msg.info.resolution
            or self.map_msg.info.origin.position.x != msg.info.origin.position.x
            or self.map_msg.info.origin.position.y != msg.info.origin.position.y
        )
        self.map_msg = msg
        if reinit:
            h, w = msg.info.height, msg.info.width
            for name in self.gases.keys():
                self.grid_mean[name] = np.full((h, w), np.nan, dtype=np.float32)
                self.grid_count[name] = np.zeros((h, w), dtype=np.uint32)
            self.get_logger().info(
                f"gas_mapper: map received, size: {w}x{h}, res: {msg.info.resolution:.3f}"
            )

    def _make_gas_cb(self, name: str):
        def cb(msg: Float32) -> None:
            v = float(msg.data)
            # Apply threshold gating on sample ingestion
            if v >= self.ppm_threshold:
                self.last_gas[name] = v
                self.last_gas_time[name] = time.time()
        return cb

    # --- Helpers ---
    def _lookup_pose_in_map(self) -> Optional[Tuple[float, float]]:
        try:
            t: TransformStamped = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, Time())
            x = t.transform.translation.x
            y = t.transform.translation.y
            return x, y
        except (LookupException, ExtrapolationException, ConnectivityException):
            return None

    def _world_to_cell(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        info = self.map_msg.info
        res = info.resolution
        ox = info.origin.position.x
        oy = info.origin.position.y
        j = int((x - ox) / res)
        i = int((y - oy) / res)
        if 0 <= i < info.height and 0 <= j < info.width:
            return i, j
        return None

    # --- Core logic ---
    def _accumulate_samples(self) -> None:
        if self.map_msg is None:
            return
        pose = self._lookup_pose_in_map()
        if pose is None:
            return
        cell = self._world_to_cell(pose[0], pose[1])
        if cell is None:
            return

        i0, j0 = cell
        r = max(0, self.kernel_radius)
        now = time.time()

        for name in self.gases.keys():
            last = self.last_gas.get(name)
            if last is None or (now - self.last_gas_time.get(name, 0.0)) > 2.0:
                continue
            grid_m = self.grid_mean.get(name)
            grid_c = self.grid_count.get(name)
            if grid_m is None or grid_c is None:
                continue
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    ii = i0 + di
                    jj = j0 + dj
                    if 0 <= ii < grid_m.shape[0] and 0 <= jj < grid_m.shape[1]:
                        n = int(grid_c[ii, jj])
                        m = float(grid_m[ii, jj])
                        v = float(last)
                        if math.isnan(m):
                            m = v
                            n = 0
                        m_new = (n * m + v) / (n + 1)
                        grid_m[ii, jj] = m_new
                        grid_c[ii, jj] = n + 1

    def _publish_gas_maps(self) -> None:
        if self.map_msg is None:
            return
        stamp = self.get_clock().now().to_msg()
        for name, cfg in self.gases.items():
            grid_m = self.grid_mean.get(name)
            if grid_m is None:
                continue
            out = OccupancyGrid()
            out.header = self.map_msg.header
            out.header.stamp = stamp
            out.info = self.map_msg.info

            h, w = grid_m.shape
            data = np.full((h, w), -1, dtype=np.int8)
            valid = ~np.isnan(grid_m)
            max_c = float(cfg.get('max', 1000.0))
            if np.any(valid):
                scaled = np.clip((grid_m[valid] / max_c) * 100.0, 0, 100).astype(np.int8)
                data[valid] = scaled
            out.data = data.flatten().tolist()
            self.pubs[name].publish(out)


def main() -> None:
    rclpy.init()
    node = GasMapper()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
