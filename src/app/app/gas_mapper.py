#!/usr/bin/env python3
"""
Gas mapper: builds separate 2D heatmaps (OccupancyGrid) per gas topic.

Subscribes to multiple ppm topics and uses TF to place measurements in a grid.
Each map is published at a fixed rate. Values are scaled to 0..100 for RViz.

Parameters:
- world_frame: 'map' (default) or 'odom'
- base_frame: 'base_link'
- resolution: 0.2  (meters per cell)
- width: 200  (cells)
- height: 200 (cells)
- origin_x: -20.0  (bottom-left X in world coords)
- origin_y: -20.0
- decay_rate: 0.0..1.0 per second (e.g., 0.01)
- pub_rate: Hz for publishing maps (default 2.0)
- topics: list of dicts [{name: 'methane', topic: 'gas/methane/ppm'}]
- ppm_threshold: global minimum ppm to plot (default PPM_THRESHOLD_MIN)
- topic_max_ppm: list of dicts [{name: 'methane', max_ppm: 10000.0}] used for scaling to 0..100

Edit the constant PPM_THRESHOLD_MIN below to change the default plot threshold.
"""

from typing import Dict, Any, List
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
import tf2_ros
from nav_msgs.msg import OccupancyGrid, MapMetaData
from std_msgs.msg import Header


# Editable default threshold (ppm) to plot points on the map.
# You can change this constant and rebuild to set your preferred minimum.
PPM_THRESHOLD_MIN = 20.0


class GasMapper(Node):
    def __init__(self) -> None:
        super().__init__('gas_mapper')

        self.world_frame = self.declare_parameter('world_frame', 'map').get_parameter_value().string_value
        self.base_frame = self.declare_parameter('base_frame', 'base_link').get_parameter_value().string_value
        self.resolution = self.declare_parameter('resolution', 0.2).get_parameter_value().double_value
        self.width = self.declare_parameter('width', 200).get_parameter_value().integer_value
        self.height = self.declare_parameter('height', 200).get_parameter_value().integer_value
        self.origin_x = self.declare_parameter('origin_x', -20.0).get_parameter_value().double_value
        self.origin_y = self.declare_parameter('origin_y', -20.0).get_parameter_value().double_value
        self.decay_rate = self.declare_parameter('decay_rate', 0.0).get_parameter_value().double_value
        self.pub_rate = self.declare_parameter('pub_rate', 2.0).get_parameter_value().double_value
        self.ppm_threshold = self.declare_parameter('ppm_threshold', PPM_THRESHOLD_MIN).get_parameter_value().double_value

        raw_topics = self.declare_parameter('topics', [
            {'name': 'methane', 'topic': 'gas/methane/ppm'},
            {'name': 'lpg', 'topic': 'gas/lpg/ppm'},
            {'name': 'co', 'topic': 'gas/co/ppm'},
            {'name': 'air_quality', 'topic': 'gas/air_quality/ppm'},
        ]).value
        self.topic_defs: List[Dict[str, Any]] = self._parse_list(raw_topics)

        # Per-gas max ppm for scaling to 0..100 in occupancy grid
        # Defaults based on typical sensor sensitivity ranges.
        raw_scales = self.declare_parameter('topic_max_ppm', [
            {'name': 'methane', 'max_ppm': 10000.0},      # MQ-4
            {'name': 'lpg', 'max_ppm': 10000.0},          # MQ-5
            {'name': 'co', 'max_ppm': 10000.0},           # MQ-7
            {'name': 'air_quality', 'max_ppm': 1000.0},   # MQ-135 (benzene/alcohol ~ up to 1000)
        ]).value
        self.topic_max: Dict[str, float] = {}
        for ent in self._parse_list(raw_scales):
            try:
                self.topic_max[str(ent['name'])] = float(ent['max_ppm'])
            except Exception:
                pass

        # Internal float grids and publishers
        self.grids: Dict[str, List[float]] = {}
        self.pubs: Dict[str, Any] = {}
        self.meta = self._build_metadata()
        for td in self.topic_defs:
            name = str(td.get('name'))
            self.grids[name] = [float('nan')] * (self.width * self.height)
            self.pubs[name] = self.create_publisher(OccupancyGrid, f'gas_map/{name}', 1)

        # TF
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=5.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subscriptions
        from std_msgs.msg import Float32
        self.last_values: Dict[str, float] = {str(td['name']): float('nan') for td in self.topic_defs}
        for td in self.topic_defs:
            topic = str(td.get('topic'))
            name = str(td.get('name'))
            self.create_subscription(Float32, topic, lambda msg, n=name: self._ppm_cb(n, msg), 10)

        # Timers: update and publish
        self.update_timer = self.create_timer(0.1, self._update)
        self.publish_timer = self.create_timer(max(0.1, 1.0 / self.pub_rate), self._publish)

    def _parse_list(self, raw):
        if isinstance(raw, list):
            if raw and isinstance(raw[0], str):
                try:
                    import yaml
                    return yaml.safe_load('\n'.join(raw))
                except Exception:
                    return []
            return raw
        if isinstance(raw, str):
            try:
                import yaml
                return yaml.safe_load(raw)
            except Exception:
                return []
        return []

    def _build_metadata(self) -> MapMetaData:
        meta = MapMetaData()
        meta.resolution = float(self.resolution)
        meta.width = int(self.width)
        meta.height = int(self.height)
        meta.origin.position.x = float(self.origin_x)
        meta.origin.position.y = float(self.origin_y)
        meta.origin.position.z = 0.0
        meta.origin.orientation.w = 1.0
        return meta

    def _ppm_cb(self, name: str, msg):
        self.last_values[name] = float(msg.data)

    def _world_to_index(self, x: float, y: float):
        ix = int((x - self.origin_x) / self.resolution)
        iy = int((y - self.origin_y) / self.resolution)
        if 0 <= ix < self.width and 0 <= iy < self.height:
            return iy * self.width + ix
        return None

    def _update(self):
        # Get robot pose
        try:
            tf = self.tf_buffer.lookup_transform(self.world_frame, self.base_frame, Time())
            x = tf.transform.translation.x
            y = tf.transform.translation.y
        except Exception:
            return

        idx = self._world_to_index(x, y)
        if idx is None:
            return

        # Simple decay
        if self.decay_rate > 0:
            decay = max(0.0, 1.0 - self.decay_rate * 0.1)
            for name, grid in self.grids.items():
                for i in range(len(grid)):
                    v = grid[i]
                    if not math.isnan(v):
                        grid[i] = v * decay

        # Update a small neighborhood (3x3) with the latest measurement
        for name, value in self.last_values.items():
            if math.isnan(value):
                continue
            # Only plot above threshold
            if value >= self.ppm_threshold:
                self._splat(name, x, y, value)

    def _splat(self, name: str, x: float, y: float, value: float):
        # Gaussian kernel over 3x3 cells centered at robot pose
        cx = int((x - self.origin_x) / self.resolution)
        cy = int((y - self.origin_y) / self.resolution)
        sigma2 = 1.0
        grid = self.grids[name]
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ix = cx + dx
                iy = cy + dy
                if 0 <= ix < self.width and 0 <= iy < self.height:
                    w = math.exp(-0.5 * ((dx * dx + dy * dy) / sigma2))
                    idx = iy * self.width + ix
                    prev = grid[idx]
                    if math.isnan(prev):
                        grid[idx] = value * w
                    else:
                        # Weighted max blend to keep peaks
                        grid[idx] = max(prev * 0.5, value * w)

    def _publish(self):
        stamp = self.get_clock().now().to_msg()
        for name, grid in self.grids.items():
            msg = OccupancyGrid()
            msg.header = Header()
            msg.header.stamp = stamp
            msg.header.frame_id = self.world_frame
            msg.info = self.meta
            # Convert to 0..100 with auto scaling using a soft cap
            data: List[int] = []
            max_ppm = float(self.topic_max.get(name, 5000.0))
            for v in grid:
                if math.isnan(v):
                    data.append(-1)
                else:
                    # Soft-clip using per-topic max ppm
                    scaled = max(0.0, min(1.0, v / max_ppm))
                    data.append(int(round(scaled * 100)))
            msg.data = data
            self.pubs[name].publish(msg)


def main() -> None:
    rclpy.init()
    node = GasMapper()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
