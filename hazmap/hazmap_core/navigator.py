"""Navigate with direct control and Nav2 fallback using LiDAR safety."""

from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Time as RosTime
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import csv
import math
import os
import time


class Navigator:
    """Wrapper around Nav2 NavigateToPose action + direct LiDAR-safe planner."""

    OBSTACLE_EMERGENCY_DIST = 0.22   # hard stop if anything this close (m)
    OBSTACLE_STOP_DIST      = 0.30   # abort forward motion (m)
    OBSTACLE_SLOW_DIST      = 0.50   # start decelerating (m)
    FRONT_ARC_DEG           = 35.0   # half-angle of the forward cone (°)

    def __init__(
        self,
        node: Node,
        pose_provider=None,
        safety_check=None,
        direct_enabled: bool = True,
        direct_linear_speed: float = 0.15,
        direct_angular_speed: float = 0.8,
        direct_xy_tolerance: float = 0.08,
        direct_yaw_tolerance: float = 0.20,
        direct_timeout: float = 20.0,
        direct_fallback_to_nav2: bool = True,
    ):
        self.node = node
        self.pose_provider = pose_provider
        self.safety_check = safety_check        # (rx,ry,tx,ty) → bool
        self.direct_enabled = direct_enabled
        self.direct_linear_speed = direct_linear_speed
        self.direct_angular_speed = direct_angular_speed
        self.direct_xy_tolerance = direct_xy_tolerance
        self.direct_yaw_tolerance = direct_yaw_tolerance
        self.direct_timeout = direct_timeout
        self.direct_fallback_to_nav2 = direct_fallback_to_nav2

        self.cmd_pub = node.create_publisher(Twist, '/cmd_vel', 10)
        self.client = ActionClient(node, NavigateToPose, 'navigate_to_pose')
        self._goal_handle = None

        self._latest_scan: LaserScan = None
        cb_group = ReentrantCallbackGroup()
        self._scan_sub = node.create_subscription(
            LaserScan, '/scan', self._scan_cb, 10,
            callback_group=cb_group,
        )

        self._traj_log = None
        self._traj_writer = None
        self._init_trajectory_log()

        self.node.get_logger().info(
            'Waiting for Nav2 navigate_to_pose action server...'
        )
        if not self.client.wait_for_server(timeout_sec=30.0):
            self.node.get_logger().error('Nav2 action server not available!')
        else:
            self.node.get_logger().info('Nav2 action server connected.')

    def _scan_cb(self, msg: LaserScan):
        self._latest_scan = msg

    def _get_front_min_distance(self) -> float:
        scan = self._latest_scan
        if scan is None:
            return float('inf')

        half_arc = math.radians(self.FRONT_ARC_DEG)
        min_dist = float('inf')

        for i in range(len(scan.ranges)):
            r = scan.ranges[i]
            if math.isinf(r) or math.isnan(r) or r < scan.range_min:
                continue
            angle = scan.angle_min + i * scan.angle_increment
            angle = math.atan2(math.sin(angle), math.cos(angle))
            if abs(angle) <= half_arc:
                if r < min_dist:
                    min_dist = r

        return min_dist

    def _init_trajectory_log(self):
        log_dir = os.path.expanduser('~')
        log_path = os.path.join(log_dir, 'hazmap_trajectory.csv')
        try:
            self._traj_log = open(log_path, 'w', newline='')
            self._traj_writer = csv.writer(self._traj_log)
            self._traj_writer.writerow([
                'time', 'robot_x', 'robot_y', 'robot_yaw',
                'target_x', 'target_y', 'method', 'result', 'note',
            ])
            self.node.get_logger().info(
                f'Trajectory log: {log_path}'
            )
        except Exception as e:
            self.node.get_logger().warn(f'Cannot open trajectory log: {e}')

    def _log_nav_event(
        self, tx: float, ty: float, method: str,
        result: str, note: str = '',
    ):
        """Append one row to the trajectory CSV."""
        if self._traj_writer is None:
            return
        try:
            rx, ry, ryaw = 0.0, 0.0, 0.0
            if self.pose_provider:
                pose = self.pose_provider()
                if pose and len(pose) == 3:
                    rx, ry, ryaw = pose
            self._traj_writer.writerow([
                f'{time.time():.3f}',
                f'{rx:.4f}', f'{ry:.4f}', f'{ryaw:.3f}',
                f'{tx:.4f}', f'{ty:.4f}',
                method, result, note,
            ])
            self._traj_log.flush()
        except Exception:
            pass

    def backup(self, distance: float = 0.25, speed: float = 0.10,
               rotate_angle: float = 0.5):
        if distance <= 0:
            return

        duration = distance / speed
        self.node.get_logger().info(
            f'Recovery: backing up {distance:.2f}m …'
        )
        t0 = time.time()
        while time.time() - t0 < duration:
            cmd = Twist()
            cmd.linear.x = -speed
            self.cmd_pub.publish(cmd)
            time.sleep(0.05)
        self._stop_robot()
        time.sleep(0.2)

        if abs(rotate_angle) < 0.01:
            return

        rot_dir = self._pick_rotation_direction()
        rot_speed = 0.6  # rad/s
        rot_duration = abs(rotate_angle) / rot_speed

        self.node.get_logger().info(
            f'Recovery: rotating {math.degrees(rotate_angle):.0f}° '
            f'{"CCW" if rot_dir > 0 else "CW"} …'
        )
        t0 = time.time()
        while time.time() - t0 < rot_duration:
            cmd = Twist()
            cmd.angular.z = rot_dir * rot_speed
            self.cmd_pub.publish(cmd)
            time.sleep(0.05)
        self._stop_robot()
        time.sleep(0.2)

    def _pick_rotation_direction(self) -> float:
        scan = self._latest_scan
        if scan is None:
            return 1.0   # default: CCW

        left_sum = 0.0
        left_cnt = 0
        right_sum = 0.0
        right_cnt = 0

        for i in range(len(scan.ranges)):
            r = scan.ranges[i]
            if math.isinf(r) or math.isnan(r) or r < scan.range_min:
                continue
            angle = scan.angle_min + i * scan.angle_increment
            angle = math.atan2(math.sin(angle), math.cos(angle))
            if 0.1 < angle < 1.5:      # left side (10°–85°)
                left_sum += r
                left_cnt += 1
            elif -1.5 < angle < -0.1:   # right side
                right_sum += r
                right_cnt += 1

        left_avg = (left_sum / left_cnt) if left_cnt > 0 else 0.0
        right_avg = (right_sum / right_cnt) if right_cnt > 0 else 0.0

        return 1.0 if left_avg >= right_avg else -1.0

    @staticmethod
    def _wait_for_future(future, timeout: float, poll: float = 0.05) -> bool:
        """Block until *future* completes or *timeout* elapses."""
        t0 = time.time()
        while not future.done():
            if time.time() - t0 > timeout:
                return False
            time.sleep(poll)
        return True

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def _stop_robot(self):
        msg = Twist()
        self.cmd_pub.publish(msg)

    def _go_to_direct(self, x: float, y: float, timeout: float) -> bool:
        if self.pose_provider is None:
            return False

        t0 = time.time()
        k_ang = 2.0          # angular gain
        decel_zone = 0.25    # start decelerating within this distance (m)
        grid_safety_interval = 3   # occupancy-grid check every N cycles
        it = 0

        obstacle_block_start = None
        OBSTACLE_BLOCK_TIMEOUT = 1.5  # abort if blocked for this long

        while time.time() - t0 < timeout:
            pose = self.pose_provider()
            if pose is None or len(pose) != 3:
                self._stop_robot()
                return False

            rx, ry, ryaw = pose
            dx = x - rx
            dy = y - ry
            dist = math.hypot(dx, dy)

            if dist <= self.direct_xy_tolerance:
                self._stop_robot()
                self._log_nav_event(x, y, 'direct', 'ok')
                return True

            it += 1

            target_heading = math.atan2(dy, dx)
            heading_err = self._normalize_angle(target_heading - ryaw)

            cmd = Twist()

            cmd.angular.z = max(
                -self.direct_angular_speed,
                min(self.direct_angular_speed, k_ang * heading_err),
            )

            heading_factor = max(0.0, math.cos(heading_err))
            dist_factor = min(1.0, dist / decel_zone)
            cmd.linear.x = self.direct_linear_speed * heading_factor * dist_factor
            if cmd.linear.x < 0.015:
                cmd.linear.x = 0.0   # deadband — pure rotation

            front_min = self._get_front_min_distance()

            if front_min < self.OBSTACLE_EMERGENCY_DIST:
                self._stop_robot()
                self.node.get_logger().warn(
                    f'Direct nav: EMERGENCY obstacle at {front_min:.2f}m — '
                    f'aborting immediately'
                )
                self._log_nav_event(
                    x, y, 'direct', 'fail',
                    f'emergency_obstacle_{front_min:.2f}m',
                )
                return False

            if front_min < self.OBSTACLE_STOP_DIST:
                cmd.linear.x = 0.0   # stop forward, still allow rotation
                if obstacle_block_start is None:
                    obstacle_block_start = time.time()
                elif time.time() - obstacle_block_start > OBSTACLE_BLOCK_TIMEOUT:
                    self._stop_robot()
                    self.node.get_logger().warn(
                        f'Direct nav: blocked for {OBSTACLE_BLOCK_TIMEOUT}s '
                        f'(front={front_min:.2f}m) — aborting'
                    )
                    self._log_nav_event(
                        x, y, 'direct', 'fail',
                        f'blocked_{front_min:.2f}m',
                    )
                    return False
            elif front_min < self.OBSTACLE_SLOW_DIST and cmd.linear.x > 0:
                slow_factor = (
                    (front_min - self.OBSTACLE_STOP_DIST)
                    / (self.OBSTACLE_SLOW_DIST - self.OBSTACLE_STOP_DIST)
                )
                slow_factor = max(0.2, min(1.0, slow_factor))
                cmd.linear.x *= slow_factor
                obstacle_block_start = None   # not blocked, just cautious
            else:
                obstacle_block_start = None   # clear

            if self.safety_check and it % grid_safety_interval == 0:
                if not self.safety_check(rx, ry, x, y):
                    self._stop_robot()
                    self.node.get_logger().warn(
                        'Direct nav: grid safety check failed — aborting'
                    )
                    self._log_nav_event(
                        x, y, 'direct', 'fail', 'grid_safety_abort',
                    )
                    return False

            self.cmd_pub.publish(cmd)
            time.sleep(0.05)

        self._stop_robot()
        self._log_nav_event(x, y, 'direct', 'fail', 'timeout')
        return False

    def _go_to_nav2(
        self, x: float, y: float, yaw: float = 0.0, timeout: float = 120.0
    ) -> bool:
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = RosTime()
        goal.pose.pose.position.x = float(x)
        goal.pose.pose.position.y = float(y)
        goal.pose.pose.position.z = 0.0
        goal.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.pose.orientation.w = math.cos(yaw / 2.0)

        self.node.get_logger().info(
            f'Nav2 fallback goal: ({x:.2f}, {y:.2f})'
        )
        send_future = self.client.send_goal_async(
            goal, feedback_callback=self._feedback_cb
        )
        if not self._wait_for_future(send_future, timeout=10.0):
            self.node.get_logger().warn('Nav2: goal send timed out')
            self._log_nav_event(x, y, 'nav2', 'fail', 'send_timeout')
            return False

        self._goal_handle = send_future.result()
        if not self._goal_handle.accepted:
            self.node.get_logger().warn(
                f'Nav2: goal ({x:.2f},{y:.2f}) rejected'
            )
            self._log_nav_event(x, y, 'nav2', 'fail', 'rejected')
            return False

        result_future = self._goal_handle.get_result_async()
        if not self._wait_for_future(result_future, timeout=timeout):
            self.node.get_logger().warn(
                f'Nav2: navigation timed out after {timeout}s'
            )
            self.cancel_navigation()
            self._log_nav_event(x, y, 'nav2', 'fail', 'timeout')
            return False

        status = result_future.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.node.get_logger().info(
                f'Nav2: arrived at ({x:.2f}, {y:.2f})'
            )
            self._log_nav_event(x, y, 'nav2', 'ok')
            return True
        else:
            self.node.get_logger().warn(
                f'Nav2: failed with status {status}'
            )
            self._log_nav_event(x, y, 'nav2', 'fail', f'status_{status}')
            return False

    def go_to(
        self,
        x: float,
        y: float,
        yaw: float = 0.0,
        timeout: float = 120.0,
        prefer_direct: bool = False,
        force_nav2: bool = False,
    ) -> bool:
        if (prefer_direct and self.direct_enabled and not force_nav2):
            self.node.get_logger().info(
                f'Direct goal: ({x:.2f}, {y:.2f})'
            )
            if self._go_to_direct(x, y, timeout=min(timeout, self.direct_timeout)):
                self.node.get_logger().info(
                    f'Arrived (direct) at ({x:.2f}, {y:.2f})'
                )
                return True
            self.node.get_logger().warn(
                f'Direct navigation failed for ({x:.2f}, {y:.2f})'
            )
            if not self.direct_fallback_to_nav2:
                return False

        return self._go_to_nav2(x, y, yaw=yaw, timeout=timeout)

    def _feedback_cb(self, feedback_msg):
        pass  # can be used for distance-remaining logging

    def cancel_navigation(self):
        if self._goal_handle is not None:
            self.node.get_logger().info('Cancelling navigation…')
            cancel_future = self._goal_handle.cancel_goal_async()
            self._wait_for_future(cancel_future, timeout=5.0)
            self._goal_handle = None

    def rotate_360(self, angular_speed: float = 0.5):
        self.node.get_logger().info('Performing 360° rotation for SLAM discovery …')

        if self.pose_provider is None:
            duration = (2 * math.pi) / angular_speed
            t0 = time.time()
            while time.time() - t0 < duration:
                cmd = Twist()
                cmd.angular.z = angular_speed
                self.cmd_pub.publish(cmd)
                time.sleep(0.05)
            self._stop_robot()
            time.sleep(0.5)  # let SLAM settle
            self.node.get_logger().info('360° rotation complete (timed).')
            return

        pose = self.pose_provider()
        if pose is None or len(pose) != 3:
            self.node.get_logger().warn('Cannot get pose for rotation.')
            return

        _, _, start_yaw = pose
        cumulative = 0.0
        prev_yaw = start_yaw
        target = 2 * math.pi  # full circle
        timeout = target / angular_speed + 5.0  # generous timeout
        t0 = time.time()

        while cumulative < target and time.time() - t0 < timeout:
            cmd = Twist()
            cmd.angular.z = angular_speed
            self.cmd_pub.publish(cmd)
            time.sleep(0.05)

            pose = self.pose_provider()
            if pose is None or len(pose) != 3:
                continue
            _, _, cur_yaw = pose
            delta = self._normalize_angle(cur_yaw - prev_yaw)
            if delta > 0:  # only count positive (same-direction) rotation
                cumulative += delta
            prev_yaw = cur_yaw

        self._stop_robot()
        time.sleep(0.5)  # let SLAM settle
        self.node.get_logger().info(
            f'360° rotation complete ({math.degrees(cumulative):.0f}° actual).'
        )

    def is_server_ready(self) -> bool:
        return self.client.server_is_ready()
