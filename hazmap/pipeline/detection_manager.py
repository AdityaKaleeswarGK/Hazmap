import math
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from visualization_msgs.msg import Marker, MarkerArray


# ── Pre-defined colours for the 5 HazMap detection classes ───────────────
_DETECTION_CLASS_COLORS: dict = {
    'coloured_spills':    (0.60, 0.00, 0.80),   # purple
    'fire':               (1.00, 0.00, 0.00),   # red
    'spills_shrouded':    (1.00, 0.55, 0.00),   # orange
    'smoke':              (0.50, 0.50, 0.50),   # gray
    'translucent_spills': (0.00, 1.00, 1.00),   # cyan
}


@dataclass
class DetectionConfig:
    """Definition of one visual detection hazard class."""
    class_name: str
    priority: int        # lower number = more severe
    color: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self):
        if self.color == (0.0, 0.0, 0.0):
            self.color = _DETECTION_CLASS_COLORS.get(
                self.class_name,
                self._generate_color(self.class_name),
            )

    @staticmethod
    def _generate_color(name: str) -> Tuple[float, float, float]:
        rng = np.random.RandomState(hash(name) % 2**31)
        return (
            float(rng.uniform(0.2, 1.0)),
            float(rng.uniform(0.2, 1.0)),
            float(rng.uniform(0.2, 1.0)),
        )


class DetectedObject:
    """Represents a unique detected object in the map."""

    def __init__(self, obj_id: int, cls_name: str,
                 map_x: float, map_y: float, map_z: float,
                 priority: int = 5):
        self.obj_id = obj_id
        self.cls_name = cls_name
        self.map_x = map_x
        self.map_y = map_y
        self.map_z = map_z
        self.priority = priority
        self.detection_count = 1
        self.alpha = 0.3  # EMA smoothing factor

    def update_position(self, new_x: float, new_y: float, new_z: float):
        """Exponential moving average update."""
        self.map_x = self.alpha * new_x + (1 - self.alpha) * self.map_x
        self.map_y = self.alpha * new_y + (1 - self.alpha) * self.map_y
        self.map_z = self.alpha * new_z + (1 - self.alpha) * self.map_z
        self.detection_count += 1


class DetectionManager:

    CONFIDENCE_THRESHOLD = 0.5
    DISTANCE_THRESHOLD = 0.5    # m — same class within this = same object
    MIN_DETECTIONS = 3           # confirm after this many sightings

    def __init__(
        self,
        node,
        model_dir: str,
        detection_configs: List[DetectionConfig],
        splash_radius: float = 0.5,
        grid_resolution: float = 0.10,
        show_display: bool = False,
    ):
        self.node = node
        self.enabled = False
        self.detection_configs = detection_configs
        self.splash_radius = splash_radius
        self.grid_resolution = grid_resolution
        self.show_display = show_display

        # Priority / colour lookup
        self.priority_map: dict = {}
        self.color_map: dict = {}
        for cfg in detection_configs:
            self.priority_map[cfg.class_name] = cfg.priority
            self.color_map[cfg.class_name] = cfg.color

        self.default_priority = (
            max((c.priority for c in detection_configs), default=10) + 1
        )

        # Detection registry
        self.detected_objects: List[DetectedObject] = []
        self.next_obj_id = 0

        # Camera intrinsics (filled by camera_info callback)
        self.fx = None
        self.fy = None
        self.cx_cam = None
        self.cy_cam = None

        # ── Find model ──────────────────────────────────────────────
        model_path = self._find_model(model_dir)
        if model_path is None:
            node.get_logger().info(
                'No YOLO model found in model/ directory — '
                'CV detection pipeline disabled.'
            )
            return

        # ── Load YOLO model ─────────────────────────────────────────
        try:
            from ultralytics import YOLO            # noqa: delayed import
            self.model = YOLO(model_path)
            node.get_logger().info(f'YOLO model loaded: {model_path}')
        except ImportError:
            node.get_logger().warn(
                'ultralytics package not installed — CV pipeline disabled.  '
                'Install with: pip install ultralytics'
            )
            return
        except Exception as exc:
            node.get_logger().warn(f'Failed to load YOLO model: {exc}')
            return

        # ── ROS interfaces ──────────────────────────────────────────
        try:
            self._setup_ros_interfaces(node)
            self.enabled = True
            node.get_logger().info(
                'HazMap CV detection pipeline ENABLED.'
            )
        except Exception as exc:
            node.get_logger().warn(
                f'Failed to set up CV pipeline ROS interfaces: {exc}'
            )

    # ─── Model discovery ────────────────────────────────────────────

    @staticmethod
    def _find_model(model_dir: str) -> Optional[str]:
        """Return path of the first model file in *model_dir*, or None."""
        if not os.path.isdir(model_dir):
            return None
        extensions = ('.pt', '.onnx', '.engine', '.torchscript')
        for fname in sorted(os.listdir(model_dir)):
            if any(fname.endswith(ext) for ext in extensions):
                return os.path.join(model_dir, fname)
        return None

    # ─── ROS setup ──────────────────────────────────────────────────

    def _setup_ros_interfaces(self, node):
        import tf2_ros
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from cv_bridge import CvBridge
        import message_filters
        from sensor_msgs.msg import Image, CameraInfo

        self.bridge = CvBridge()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, node)

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)

        self.camera_info_sub = node.create_subscription(
            CameraInfo, '/camera/color/camera_info',
            self._camera_info_cb, qos,
        )

        self.marker_pub = node.create_publisher(
            MarkerArray, '/hazmap/detections', 10,
        )

        rgb_sub = message_filters.Subscriber(
            node, Image, '/camera/color/image_raw', qos_profile=qos,
        )
        depth_sub = message_filters.Subscriber(
            node, Image, '/camera/depth/image_raw', qos_profile=qos,
        )

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub], queue_size=20, slop=0.5,
        )
        self.sync.registerCallback(self._synced_callback)

    # ─── Camera helpers ─────────────────────────────────────────────

    def _camera_info_cb(self, msg):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx_cam = msg.k[2]
        self.cy_cam = msg.k[5]
        self.node.get_logger().info(
            f'Camera intrinsics: fx={self.fx:.2f} fy={self.fy:.2f} '
            f'cx={self.cx_cam:.2f} cy={self.cy_cam:.2f}'
        )
        self.node.destroy_subscription(self.camera_info_sub)

    @staticmethod
    def _get_region_depth(depth_frame, cx_int, cy_int, region=5):
        y_min = max(0, cy_int - region)
        y_max = min(depth_frame.shape[0], cy_int + region)
        x_min = max(0, cx_int - region)
        x_max = min(depth_frame.shape[1], cx_int + region)
        patch = depth_frame[y_min:y_max, x_min:x_max]
        valid = patch[patch > 0]
        return float(valid.mean()) if len(valid) > 0 else 0.0

    def _transform_to_map(self, x, y, z, stamp):
        import rclpy
        from geometry_msgs.msg import PointStamped
        from tf2_geometry_msgs import do_transform_point

        pt = PointStamped()
        pt.header.frame_id = 'camera_color_optical_frame'
        pt.header.stamp = stamp
        pt.point.x, pt.point.y, pt.point.z = x, y, z

        try:
            tf = self.tf_buffer.lookup_transform(
                'map', 'camera_color_optical_frame', stamp,
                timeout=rclpy.duration.Duration(seconds=0.1),
            )
            out = do_transform_point(pt, tf)
            return out.point.x, out.point.y, out.point.z
        except Exception as exc:
            self.node.get_logger().warn(
                f'TF transform failed: {exc}',
                throttle_duration_sec=2.0,
            )
            return None

    # ─── Detection registry ─────────────────────────────────────────

    def _find_matching_object(self, cls_name, mx, my, mz):
        for obj in self.detected_objects:
            if obj.cls_name != cls_name:
                continue
            d = math.sqrt(
                (obj.map_x - mx) ** 2
                + (obj.map_y - my) ** 2
                + (obj.map_z - mz) ** 2
            )
            if d < self.DISTANCE_THRESHOLD:
                return obj
        return None

    def _get_priority(self, cls_name: str) -> int:
        return self.priority_map.get(cls_name, self.default_priority)

    def _get_class_color(self, cls_name: str) -> Tuple[float, float, float]:
        if cls_name in self.color_map:
            return self.color_map[cls_name]
        rng = np.random.RandomState(hash(cls_name) % 2**31)
        color = (
            float(rng.uniform(0.2, 1.0)),
            float(rng.uniform(0.2, 1.0)),
            float(rng.uniform(0.2, 1.0)),
        )
        self.color_map[cls_name] = color
        return color

    # ─── Main detection callback ────────────────────────────────────

    def _synced_callback(self, rgb_msg, depth_msg):
        """Process synchronised RGB + Depth frames through YOLO."""
        import cv2

        frame = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
        depth = self.bridge.imgmsg_to_cv2(
            depth_msg, desired_encoding='passthrough',
        )

        results = self.model(frame, verbose=False)

        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < self.CONFIDENCE_THRESHOLD:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]
                cx, cy = float(box.xywh[0][0]), float(box.xywh[0][1])
                cx_int, cy_int = int(cx), int(cy)

                depth_val = self._get_region_depth(depth, cx_int, cy_int)
                depth_m = depth_val / 1000.0
                if depth_m <= 0.0 or self.fx is None:
                    continue

                X_cam = (cx - self.cx_cam) * depth_m / self.fx
                Y_cam = (cy - self.cy_cam) * depth_m / self.fy
                Z_cam = depth_m

                coords = self._transform_to_map(
                    X_cam, Y_cam, Z_cam, rgb_msg.header.stamp,
                )
                if coords is None:
                    continue
                map_x, map_y, map_z = coords
                priority = self._get_priority(cls_name)

                existing = self._find_matching_object(
                    cls_name, map_x, map_y, map_z,
                )
                if existing:
                    existing.update_position(map_x, map_y, map_z)
                else:
                    self.detected_objects.append(
                        DetectedObject(
                            self.next_obj_id, cls_name,
                            map_x, map_y, map_z, priority,
                        )
                    )
                    self.next_obj_id += 1

                if self.show_display:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    lbl = f'{cls_name} {conf:.2f} | Z:{Z_cam:.2f}m'
                    cv2.putText(
                        frame, lbl, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                    )

        self._publish_markers()

        if self.show_display:
            confirmed = sum(
                1 for o in self.detected_objects
                if o.detection_count >= self.MIN_DETECTIONS
            )
            cv2.putText(
                frame, f'Detections: {confirmed}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2,
            )
            cv2.imshow('HazMap Detection', frame)
            cv2.waitKey(1)

    # ─── Marker publishing ──────────────────────────────────────────

    def _publish_markers(self):
        marker_array = MarkerArray()

        for obj in self.detected_objects:
            if obj.detection_count < self.MIN_DETECTIONS:
                continue
            r, g, b = self._get_class_color(obj.cls_name)

            # Sphere pin
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = self.node.get_clock().now().to_msg()
            m.ns = 'hazmap_detections'
            m.id = obj.obj_id
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = obj.map_x
            m.pose.position.y = obj.map_y
            m.pose.position.z = obj.map_z
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.2
            m.color.r, m.color.g, m.color.b = float(r), float(g), float(b)
            m.color.a = 1.0
            m.lifetime.sec = 0
            marker_array.markers.append(m)

            # Text label
            t = Marker()
            t.header.frame_id = 'map'
            t.header.stamp = self.node.get_clock().now().to_msg()
            t.ns = 'hazmap_labels'
            t.id = obj.obj_id
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position.x = obj.map_x
            t.pose.position.y = obj.map_y
            t.pose.position.z = obj.map_z + 0.3
            t.pose.orientation.w = 1.0
            t.scale.z = 0.15
            t.color.r = t.color.g = t.color.b = 1.0
            t.color.a = 1.0
            t.text = f'{obj.cls_name} ({obj.detection_count})'
            t.lifetime.sec = 0
            marker_array.markers.append(t)

        if marker_array.markers:
            self.marker_pub.publish(marker_array)

    # ─── Data access for saving ─────────────────────────────────────

    def get_confirmed_detections(self) -> List[DetectedObject]:
        """Return only detections that have been seen >= MIN_DETECTIONS times."""
        return [
            o for o in self.detected_objects
            if o.detection_count >= self.MIN_DETECTIONS
        ]

    def get_detection_impact_grid(
        self,
        x_min: float, y_min: float,
        x_max: float, y_max: float,
    ) -> Optional[np.ndarray]:
        confirmed = self.get_confirmed_detections()
        if not confirmed:
            return None

        cols = max(1, int((x_max - x_min) / self.grid_resolution))
        rows = max(1, int((y_max - y_min) / self.grid_resolution))
        grid = np.zeros((rows, cols), dtype=np.float64)

        splash_cells = max(1, int(self.splash_radius / self.grid_resolution))

        for det in confirmed:
            cc = int((det.map_x - x_min) / self.grid_resolution)
            cr = int((det.map_y - y_min) / self.grid_resolution)
            weight = 1.0 / det.priority
            intensity = weight * min(det.detection_count / 10.0, 1.0)

            for dr in range(-splash_cells, splash_cells + 1):
                for dc in range(-splash_cells, splash_cells + 1):
                    r = cr + dr
                    c = cc + dc
                    if not (0 <= r < rows and 0 <= c < cols):
                        continue
                    dist = math.sqrt(dr * dr + dc * dc) * self.grid_resolution
                    if dist > self.splash_radius:
                        continue
                    falloff = math.exp(
                        -dist * dist / (2.0 * (self.splash_radius * 0.5) ** 2)
                    )
                    grid[r, c] = max(grid[r, c], intensity * falloff)

        gmax = float(np.max(grid))
        if gmax > 1e-12:
            grid /= gmax
        return grid


# ─── Parameter loader ───────────────────────────────────────────────────

def load_detection_configs_from_params(node) -> List[DetectionConfig]:
    """Read detection class definitions from ROS2 parameters."""
    node.declare_parameter('detection_class_names', [
        'coloured_spills', 'fire', 'spills_shrouded',
        'smoke', 'translucent_spills',
    ])
    node.declare_parameter('detection_priorities', [5, 1, 3, 2, 4])

    names = node.get_parameter('detection_class_names').value
    priorities = node.get_parameter('detection_priorities').value

    configs: List[DetectionConfig] = []
    for i, name in enumerate(names):
        pri = priorities[i] if i < len(priorities) else (i + 1)
        color = _DETECTION_CLASS_COLORS.get(
            name, DetectionConfig._generate_color(name),
        )
        configs.append(DetectionConfig(
            class_name=name, priority=pri, color=color,
        ))

    return configs

