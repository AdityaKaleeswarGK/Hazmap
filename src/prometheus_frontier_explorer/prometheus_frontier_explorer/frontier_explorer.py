#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from visualization_msgs.msg import MarkerArray, Marker
import math
import tf2_ros
from geometry_msgs.msg import TransformStamped, PoseStamped
from nav2_msgs.action import NavigateToPose


class NearestFrontier(Node):
    def __init__(self):
        super().__init__('frontier_explorer')

        # Subscription to frontier centroids
        self.centroid_sub = self.create_subscription(
            MarkerArray,
            '/frontier_centroids',
            self.centroid_callback,
            10
        )

        # Publisher for nearest cluster marker (just for visualization)
        self.nearest_marker_pub = self.create_publisher(Marker, '/nearest_frontier', 10)

        # TF buffer and listener for map -> base_link
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Action client for Nav2
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Minimum distance threshold for clusters
        self.min_distance = 1.5

        # Flag to track navigation state
        self.navigating = False

        self.get_logger().info("Nearest Frontier Node started (with Nav2)")

    def centroid_callback(self, msg: MarkerArray):
        # Do nothing if already navigating
        if self.navigating:
            return

        # Get robot pose in map frame
        try:
            t: TransformStamped = self.tf_buffer.lookup_transform(
                'map',       # target frame
                'base_link', # source frame
                rclpy.time.Time()
            )
            robot_x = t.transform.translation.x
            robot_y = t.transform.translation.y
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return

        # Extract centroids from MarkerArray
        centroids = [(m.pose.position.x, m.pose.position.y) for m in msg.markers]

        if not centroids:
            self.get_logger().info("No frontier centroids available")
            return

        # Find nearest cluster beyond min_distance
        nearest_idx = -1
        nearest_dist = float('inf')

        for idx, (cx, cy) in enumerate(centroids):
            dist = math.hypot(cx - robot_x, cy - robot_y)
            if self.min_distance < dist < nearest_dist:
                nearest_dist = dist
                nearest_idx = idx

        if nearest_idx == -1:
            self.get_logger().info("No valid nearest cluster found")
            return

        nearest_cluster = centroids[nearest_idx]
        self.get_logger().info(f"Sending Nav2 goal: x={nearest_cluster[0]:.2f}, y={nearest_cluster[1]:.2f}")

        # Publish marker for visualization
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'nearest_frontier'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = nearest_cluster[0]
        marker.pose.position.y = nearest_cluster[1]
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.15
        marker.scale.y = 0.15
        marker.scale.z = 0.15
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        self.nearest_marker_pub.publish(marker)

        # Send goal to Nav2
        self.send_goal(nearest_cluster[0], nearest_cluster[1])

    def send_goal(self, x, y):
        # Create PoseStamped goal
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.orientation.w = 1.0

        # Send goal
        self.navigating = True
        self.nav_to_pose_client.wait_for_server()
        send_future = self.nav_to_pose_client.send_goal_async(goal_msg)
        send_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Goal rejected')
            self.navigating = False
            return

        self.get_logger().info('Goal accepted by Nav2')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Goal finished with result: {result}')
        self.navigating = False  # Ready for next goal


def main(args=None):
    rclpy.init(args=args)
    node = NearestFrontier()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
