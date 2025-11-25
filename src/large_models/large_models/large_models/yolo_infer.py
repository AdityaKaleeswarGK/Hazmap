#!/usr/bin/env python3
# encoding: utf-8
# ROS2 YOLO inference subscriber for RGB camera
import os
import queue
import threading
from typing import Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

try:
    from ultralytics import YOLO
except Exception as e:  # pragma: no cover
    YOLO = None


class YOLOInfer(Node):
    def __init__(self, name: str = 'yolo_infer'):
        # Follow existing package pattern: call rclpy.init() in constructor
        rclpy.init()
        super().__init__(name)

        # Parameters
        # Default camera topic matches repo launch files
        self.declare_parameter('camera_topic', '/depth_cam/rgb/image_raw')
        self.declare_parameter('model_path', '')
        self.declare_parameter('conf', 0.25)

        self.bridge = CvBridge()
        self.image_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=2)
        self.running = True
        self.window_name = 'yolo_infer'

        camera_topic: str = self.get_parameter('camera_topic').value
        user_model_path: str = self.get_parameter('model_path').value
        self.conf: float = float(self.get_parameter('conf').value)

        model_path = self._resolve_model_path(user_model_path)
        self.get_logger().info(f'Loading YOLO model: {model_path}')

        if YOLO is None:
            self.get_logger().error('Ultralytics not available. Please ensure the "ultralytics" package is installed.')
            raise RuntimeError('Ultralytics not available')

        try:
            self.model = YOLO(model_path)
            self.get_logger().info(f'Model loaded. Classes: {self.model.names}')
        except Exception as e:  # pragma: no cover
            self.get_logger().error(f'Failed to load model from {model_path}: {e}')
            raise

        # Publishers / Subscribers
        # Publish annotated image on a private topic under the node namespace
        self.result_pub = self.create_publisher(Image, '~/image_result', 1)
        # Subscribe to camera
        self.create_subscription(Image, camera_topic, self.image_callback, 1)

        # Start processing thread
        self.worker = threading.Thread(target=self._process_loop, daemon=True)
        self.worker.start()
        self.get_logger().info('YOLO inference node started.')

    def _resolve_model_path(self, user_model_path: Optional[str]) -> str:
        if user_model_path and os.path.exists(user_model_path):
            return user_model_path
        # Fallback to installed share/large_models/models/best.pt
        try:
            share_dir = get_package_share_directory('large_models')
            default_path = os.path.join(share_dir, 'models', 'best.pt')
            return default_path
        except Exception:
            # As a last resort, try local source tree path (useful in dev without install)
            local_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best.pt')
            return os.path.abspath(local_path)

    def image_callback(self, msg: Image) -> None:
        # Convert incoming image to BGR OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        frame = np.asarray(cv_image, dtype=np.uint8)
        if self.image_queue.full():
            # Drop oldest
            try:
                self.image_queue.get_nowait()
            except queue.Empty:
                pass
        self.image_queue.put(frame)

    def _process_loop(self) -> None:
        try:
            while self.running:
                try:
                    frame = self.image_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                # Run inference
                try:
                    results = self.model.predict(frame, conf=self.conf, verbose=False)
                    annotated = results[0].plot() if results else frame
                except Exception as e:  # pragma: no cover
                    self.get_logger().error(f'YOLO inference error: {e}')
                    annotated = frame

                # Publish annotated image
                out_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
                self.result_pub.publish(out_msg)

                # Optional: display window if available
                try:
                    cv2.imshow(self.window_name, annotated)
                    key = cv2.waitKey(1)
                    if key in (ord('q'), 27):
                        self.running = False
                        break
                except Exception:
                    # Headless environment: ignore GUI errors
                    pass
        finally:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass


def main():  # pragma: no cover
    node = YOLOInfer('yolo_infer')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.running = False
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()


if __name__ == '__main__':
    main()
