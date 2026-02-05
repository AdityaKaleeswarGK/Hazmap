import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy
import cv2
import time
import numpy as np
from ultralytics import YOLO
import message_filters
import tf2_ros
from geometry_msgs.msg import PointStamped
from tf2_geometry_msgs import do_transform_point

class DetectedObject:
    """Represents a unique detected object in the map."""
    def __init__(self, obj_id, cls_name, map_x, map_y, map_z):
        self.obj_id = obj_id
        self.cls_name = cls_name
        self.map_x = map_x
        self.map_y = map_y
        self.map_z = map_z
        self.detection_count = 1
        self.alpha = 0.3  # EMA smoothing factor

    def update_position(self, new_x, new_y, new_z):
        """Exponential moving average update."""
        self.map_x = self.alpha * new_x + (1 - self.alpha) * self.map_x
        self.map_y = self.alpha * new_y + (1 - self.alpha) * self.map_y
        self.map_z = self.alpha * new_z + (1 - self.alpha) * self.map_z
        self.detection_count += 1


class CameraSubscriber(Node):
    def __init__(self):
        super().__init__('camera_subscriber')
        self.bridge = CvBridge()
        self.model = YOLO('/home/gk/ros2_ws/src/hazmap/yolov8n.pt')
        self.prev_time = time.time()

        self.fx = None
        self.fy = None
        self.cx_cam = None
        self.cy_cam = None

        # Detection registry
        self.detected_objects = []
        self.next_obj_id = 0
        self.DISTANCE_THRESHOLD = 0.5    # meters — same class within this = same object
        self.CONFIDENCE_THRESHOLD = 0.5  # ignore detections below this
        self.MIN_DETECTIONS = 3          # need this many sightings before publishing pin

        # TF2 for camera → map transform
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Color map for different classes
        self.class_colors = {}

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/color/camera_info',
            self.camera_info_callback,
            qos_profile
        )

        self.marker_pub = self.create_publisher(MarkerArray, '/hazmap/detections', 10)

        # Synced RGB + Depth
        rgb_sub = message_filters.Subscriber(
            self, Image, '/camera/color/image_raw', qos_profile=qos_profile
        )
        depth_sub = message_filters.Subscriber(
            self, Image, '/camera/depth/image_raw', qos_profile=qos_profile
        )

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub], queue_size=20, slop=0.5
        )
        self.sync.registerCallback(self.synced_callback)

        self.get_logger().info("HazMap detection node started")

    def camera_info_callback(self, msg):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx_cam = msg.k[2]
        self.cy_cam = msg.k[5]
        self.get_logger().info(
            f"Camera intrinsics loaded: fx={self.fx:.2f} fy={self.fy:.2f} "
            f"cx={self.cx_cam:.2f} cy={self.cy_cam:.2f}"
        )
        self.destroy_subscription(self.camera_info_sub)

    def get_region_depth(self, depth_frame, cx_int, cy_int, region=5):
        y_min = max(0, cy_int - region)
        y_max = min(depth_frame.shape[0], cy_int + region)
        x_min = max(0, cx_int - region)
        x_max = min(depth_frame.shape[1], cx_int + region)

        depth_region = depth_frame[y_min:y_max, x_min:x_max]
        valid_depths = depth_region[depth_region > 0]

        if len(valid_depths) > 0:
            return float(valid_depths.mean())
        return 0.0

    def transform_to_map(self, x, y, z, stamp):
        """Transform a point from camera frame to map frame using TF2."""
        point = PointStamped()
        point.header.frame_id = 'camera_color_optical_frame'
        point.header.stamp = stamp
        point.point.x = x
        point.point.y = y
        point.point.z = z

        try:
            transform = self.tf_buffer.lookup_transform(
                'map',
                'camera_color_optical_frame',
                stamp,
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            transformed = do_transform_point(point, transform)
            return transformed.point.x, transformed.point.y, transformed.point.z
        except Exception as e:
            self.get_logger().warn(f"TF transform failed: {e}", throttle_duration_sec=2.0)
            return None

    def find_matching_object(self, cls_name, map_x, map_y, map_z):
        """Find an existing object of the same class within distance threshold."""
        for obj in self.detected_objects:
            if obj.cls_name != cls_name:
                continue
            dist = np.sqrt(
                (obj.map_x - map_x) ** 2 +
                (obj.map_y - map_y) ** 2 +
                (obj.map_z - map_z) ** 2
            )
            if dist < self.DISTANCE_THRESHOLD:
                return obj
        return None

    def get_class_color(self, cls_name):
        """Assign a consistent color to each class."""
        if cls_name not in self.class_colors:
            np.random.seed(hash(cls_name) % 2**32)
            self.class_colors[cls_name] = (
                np.random.uniform(0.2, 1.0),
                np.random.uniform(0.2, 1.0),
                np.random.uniform(0.2, 1.0)
            )
        return self.class_colors[cls_name]

    def publish_markers(self):
        """Publish all confirmed detections as RViz markers."""
        marker_array = MarkerArray()

        for obj in self.detected_objects:
            if obj.detection_count < self.MIN_DETECTIONS:
                continue

            r, g, b = self.get_class_color(obj.cls_name)

            # Sphere marker for the pin
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'hazmap_detections'
            marker.id = obj.obj_id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = obj.map_x
            marker.pose.position.y = obj.map_y
            marker.pose.position.z = obj.map_z
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.r = float(r)
            marker.color.g = float(g)
            marker.color.b = float(b)
            marker.color.a = 1.0
            marker.lifetime.sec = 0  # persistent
            marker_array.markers.append(marker)

            # Text label above the sphere
            text_marker = Marker()
            text_marker.header.frame_id = 'map'
            text_marker.header.stamp = self.get_clock().now().to_msg()
            text_marker.ns = 'hazmap_labels'
            text_marker.id = obj.obj_id
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = obj.map_x
            text_marker.pose.position.y = obj.map_y
            text_marker.pose.position.z = obj.map_z + 0.3
            text_marker.pose.orientation.w = 1.0
            text_marker.scale.z = 0.15
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            text_marker.text = f"{obj.cls_name} ({obj.detection_count})"
            text_marker.lifetime.sec = 0
            marker_array.markers.append(text_marker)

        if marker_array.markers:
            self.marker_pub.publish(marker_array)

    def synced_callback(self, rgb_msg, depth_msg):
        frame = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
        depth_frame = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

        results = self.model(frame, verbose=False)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = float(box.conf[0])

                # Skip low confidence detections
                if conf < self.CONFIDENCE_THRESHOLD:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]
                cx, cy, w, h = map(float, box.xywh[0])

                cx_int, cy_int = int(cx), int(cy)

                # Region-averaged depth
                depth_val = self.get_region_depth(depth_frame, cx_int, cy_int, region=5)
                depth_m = depth_val / 1000.0

                # Skip if no valid depth
                if depth_m <= 0.0:
                    # Still draw bbox but skip mapping
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{cls_name} {conf:.2f} | no depth"
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    continue

                # Pixel to camera frame 3D
                if self.fx is None:
                    continue
                X_cam = (cx - self.cx_cam) * depth_m / self.fx
                Y_cam = (cy - self.cy_cam) * depth_m / self.fy
                Z_cam = depth_m

                # Transform to map frame
                map_coords = self.transform_to_map(
                    X_cam, Y_cam, Z_cam, rgb_msg.header.stamp
                )

                if map_coords is None:
                    continue

                map_x, map_y, map_z = map_coords

                # Check registry for existing match
                existing = self.find_matching_object(cls_name, map_x, map_y, map_z)
                if existing:
                    existing.update_position(map_x, map_y, map_z)
                    status = f"UPDATED (seen {existing.detection_count}x)"
                else:
                    new_obj = DetectedObject(
                        self.next_obj_id, cls_name, map_x, map_y, map_z
                    )
                    self.detected_objects.append(new_obj)
                    self.next_obj_id += 1
                    status = "NEW"

                # Draw on frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{cls_name} {conf:.2f} | Z:{Z_cam:.2f}m"
                coord_label = f"Map: ({map_x:.2f}, {map_y:.2f}) [{status}]"
                cv2.putText(frame, label, (x1, y1 - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, coord_label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.circle(frame, (cx_int, cy_int), 5, (0, 0, 255), -1)

                self.get_logger().info(
                    f"{status}: {cls_name} | conf: {conf:.2f} | "
                    f"Map: x={map_x:.2f} y={map_y:.2f} z={map_z:.2f}"
                )

        # Publish markers to RViz
        self.publish_markers()

        # FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - self.prev_time)
        self.prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Registry count
        confirmed = sum(1 for o in self.detected_objects if o.detection_count >= self.MIN_DETECTIONS)
        cv2.putText(frame, f"Objects: {confirmed}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow("HazMap Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.get_logger().info("Stopping camera subscriber")
            rclpy.shutdown()


def main():
    rclpy.init()
    node = CameraSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()