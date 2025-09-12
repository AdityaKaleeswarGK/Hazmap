#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.timer import Timer
import os
import subprocess
from datetime import datetime
from slam_toolbox.srv import SaveMap, SerializePoseGraph

class MapAutoSaver(Node):
    def __init__(self):
        super().__init__('map_auto_saver')
        self.declare_parameter('save_path', '/tmp')
        self.declare_parameter('base_name', 'slam_map')
        self.declare_parameter('delay_sec', 5.0)
        self.declare_parameter('save_pose_graph', True)
        self.declare_parameter('make_png', True)
        self.declare_parameter('make_pdf', False)
        self.declare_parameter('once', True)

        self.save_path = self.get_parameter('save_path').value
        self.base_name = self.get_parameter('base_name').value
        self.delay_sec = float(self.get_parameter('delay_sec').value)
        self.save_pose_graph = bool(self.get_parameter('save_pose_graph').value)
        self.make_png = bool(self.get_parameter('make_png').value)
        self.make_pdf = bool(self.get_parameter('make_pdf').value)
        self.once = bool(self.get_parameter('once').value)

        os.makedirs(self.save_path, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.map_base = os.path.join(self.save_path, f"{self.base_name}_{timestamp}")
        self.get_logger().info(f"MapAutoSaver will save to: {self.map_base}")

        self._client_save = self.create_client(SaveMap, '/slam_toolbox/save_map')
        self._client_serial = self.create_client(SerializePoseGraph, '/slam_toolbox/serialize_map')

        self._timer: Timer = self.create_timer(self.delay_sec, self._on_timer)
        self._fired = False

    def _wait_for_service(self, client, name):
        if not client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn(f"Service {name} not available yet")
            return False
        return True

    def _call_save(self):
        from slam_toolbox.srv import SaveMap
        req = SaveMap.Request()
        req.name = self.map_base
        future = self._client_save.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        if future.result() is None:
            self.get_logger().error('SaveMap call failed or timed out')
            return False
        self.get_logger().info('Occupancy map saved')
        return True

    def _call_serialize(self):
        if not self.save_pose_graph:
            return
        from slam_toolbox.srv import SerializePoseGraph
        req = SerializePoseGraph.Request()
        req.filename = self.map_base + '_graph'
        future = self._client_serial.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        if future.result() is None:
            self.get_logger().warn('SerializePoseGraph failed or timed out')
        else:
            self.get_logger().info('Pose graph serialized')

    def _convert_images(self):
        pgm = self.map_base + '.pgm'
        if not os.path.exists(pgm):
            self.get_logger().warn(f"PGM file not found yet: {pgm}")
            return
        if self.make_png:
            png = self.map_base + '.png'
            if self._run_convert(pgm, png):
                self.get_logger().info(f"Created PNG: {png}")
        if self.make_pdf:
            pdf = self.map_base + '.pdf'
            if self._run_convert(pgm, pdf):
                self.get_logger().info(f"Created PDF: {pdf}")

    def _run_convert(self, src, dst):
        # Try ImageMagick 'magick' first, fall back to 'convert'
        for cmd in [['magick', 'convert', src, dst], ['convert', src, dst]]:
            try:
                r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
                if r.returncode == 0:
                    return True
            except FileNotFoundError:
                continue
            except Exception as e:
                self.get_logger().warn(f"Conversion error: {e}")
        self.get_logger().warn('No image conversion tool available (install ImageMagick)')
        return False

    def _on_timer(self):
        if self.once and self._fired:
            return
        if not self._wait_for_service(self._client_save, '/slam_toolbox/save_map'):
            return
        if self.save_pose_graph and not self._wait_for_service(self._client_serial, '/slam_toolbox/serialize_map'):
            return
        if self._call_save():
            self._call_serialize()
            self._convert_images()
            self._fired = True
            if self.once:
                self._timer.cancel()
                self.get_logger().info('MapAutoSaver finished (once=True)')


def main():
    rclpy.init()
    node = MapAutoSaver()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
