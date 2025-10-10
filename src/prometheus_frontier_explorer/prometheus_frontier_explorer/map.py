#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
from scipy.ndimage import convolve, label
import hashlib

class FrontierVisualizer(Node):
    def __init__(self):
        super().__init__('frontier_visualizer')
        
        # Map parameters
        self.width = 0
        self.height = 0
        self.resolution = 0.05
        self.origin = None
        
        # Map data
        self.map_data = None
        self.x_grid = None
        self.y_grid = None
        self.is_frontier = None
        self.last_map_hash = None  # To track changes
        
        # Subscribe to map topic
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )
        
        # Publishers for visualization markers
        self.frontier_pub = self.create_publisher(MarkerArray, '/frontier_markers', 10)
        self.centroid_pub = self.create_publisher(MarkerArray, '/frontier_centroids', 10)
        
        self.get_logger().info("Frontier Visualizer Node started")

    def map_callback(self, msg):
        """Process incoming occupancy grid messages"""
        self.width = msg.info.width
        self.height = msg.info.height
        self.resolution = msg.info.resolution
        self.origin = msg.info.origin.position

        # Convert map data
        self.map_data = np.array(msg.data, dtype=np.int8).reshape((self.height, self.width))


        # Coordinate grid
        x_coords = np.arange(self.width) * self.resolution + self.origin.x
        y_coords = np.arange(self.height) * self.resolution + self.origin.y
        self.x_grid, self.y_grid = np.meshgrid(x_coords, y_coords)

        # Detect frontiers
        self.detect_frontiers()

    def detect_frontiers(self):
        free_cells = (self.map_data == 0)
        unknown_cells = (self.map_data == -1)

        # 8-neighborhood kernel
        kernel = np.ones((3, 3), dtype=int)
        kernel[1, 1] = 0

        # Count unknown neighbors
        unknown_count = convolve(unknown_cells.astype(int), kernel, mode='constant', cval=0)

        # Only consider free cells with majority unknown neighbors (>=5)
        self.is_frontier = free_cells & (unknown_count >= 5)

        frontier_count = np.sum(self.is_frontier)
        self.get_logger().info(f'Found {frontier_count} frontier cells')

        # Visualize individual frontier cells
        self.visualize_frontiers()

        # Compute and visualize cluster centroids
        self.compute_frontier_clusters()

    def visualize_frontiers(self, limit=2000):
        """Publish frontier cells as RViz markers"""
        marker_array = MarkerArray()
        frontier_indices = np.argwhere(self.is_frontier)
        count = min(len(frontier_indices), limit)

        for i, (row, col) in enumerate(frontier_indices[:count]):
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'frontier_cells'
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.scale.z = 0.05  # Set non-zero for visibility in RViz

            marker.color.r = 0.0
            marker.color.g = 0.8
            marker.color.b = 1.0
            marker.color.a = 1.0

            marker.pose.position.x = float(self.x_grid[row, col])
            marker.pose.position.y = float(self.y_grid[row, col])
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0

            marker_array.markers.append(marker)

        self.frontier_pub.publish(marker_array)

    def compute_frontier_clusters(self):
        """Cluster frontier points and publish their centroids as markers"""
        labeled_array, num_features = label(self.is_frontier)
        centroids = []

        marker_array = MarkerArray()

        for cluster_id in range(1, num_features + 1):
            positions = np.argwhere(labeled_array == cluster_id)
            x_coords = self.x_grid[positions[:,0], positions[:,1]]
            y_coords = self.y_grid[positions[:,0], positions[:,1]]
            centroid_x = np.mean(x_coords)
            centroid_y = np.mean(y_coords)
            centroids.append((centroid_x, centroid_y))

            # Create a marker for each centroid
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'frontier_centroids'
            marker.id = cluster_id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1  # Larger sphere for centroid visibility

            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0

            marker.pose.position.x = centroid_x
            marker.pose.position.y = centroid_y
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0

            marker_array.markers.append(marker)

        self.centroid_pub.publish(marker_array)
        self.get_logger().info(f'Published {len(centroids)} frontier cluster centroids')

def main(args=None):
    rclpy.init(args=args)
    node = FrontierVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
