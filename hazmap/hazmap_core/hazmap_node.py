import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PoseStamped
from std_srvs.srv import Trigger
import math
import os
import time
import threading

from .map_manager import MapManager
from .sampling import ProgressiveSampler
from .rcg import RCG, NodeState
from .waypoint_selector import WaypointSelector
from .navigator import Navigator
from .utils import is_collision_free
from ..pipeline.sensor_manager import SensorManager, load_sensor_configs_from_params
from ..pipeline.detection_manager import DetectionManager, load_detection_configs_from_params


class HazMapNode(Node):
    """Main ROS2 node for HazMap coverage path planning."""

    def __init__(self):
        super().__init__('hazmap_node')

        self.callback_group = ReentrantCallbackGroup()

        self.declare_parameter('w', 0.35)
        self.declare_parameter('delta', 1)
        self.declare_parameter('sweep_direction', 'x')
        self.declare_parameter('obstacle_buffer', 0.18)
        self.declare_parameter('visualization_rate', 2.0)
        self.declare_parameter('frontier_min_cells', 5)
        self.declare_parameter('debug_logging', False)
        self.declare_parameter('use_hybrid_navigation', True)
        self.declare_parameter('direct_nav_same_lap_only', True)
        self.declare_parameter('direct_nav_max_distance', 0.80)
        self.declare_parameter('direct_nav_linear_speed', 0.12)
        self.declare_parameter('direct_nav_angular_speed', 0.80)
        self.declare_parameter('direct_nav_xy_tolerance', 0.08)
        self.declare_parameter('direct_nav_yaw_tolerance', 0.20)
        self.declare_parameter('direct_nav_timeout', 20.0)
        self.declare_parameter('direct_nav_fallback_to_nav2', True)
        self.declare_parameter('direct_nav_min_clearance', 0.20)
        self.w = self.get_parameter('w').value
        self.delta = self.get_parameter('delta').value
        self.sweep_direction = self.get_parameter('sweep_direction').value
        self.obstacle_buffer = self.get_parameter('obstacle_buffer').value
        self.viz_rate = self.get_parameter('visualization_rate').value
        self.frontier_min_cells = self.get_parameter('frontier_min_cells').value
        self.debug = self.get_parameter('debug_logging').value
        self.use_hybrid_navigation = self.get_parameter(
            'use_hybrid_navigation'
        ).value
        self.direct_nav_same_lap_only = self.get_parameter(
            'direct_nav_same_lap_only'
        ).value
        self.direct_nav_max_distance = self.get_parameter(
            'direct_nav_max_distance'
        ).value
        self.direct_nav_linear_speed = self.get_parameter(
            'direct_nav_linear_speed'
        ).value
        self.direct_nav_angular_speed = self.get_parameter(
            'direct_nav_angular_speed'
        ).value
        self.direct_nav_xy_tolerance = self.get_parameter(
            'direct_nav_xy_tolerance'
        ).value
        self.direct_nav_yaw_tolerance = self.get_parameter(
            'direct_nav_yaw_tolerance'
        ).value
        self.direct_nav_timeout = self.get_parameter(
            'direct_nav_timeout'
        ).value
        self.direct_nav_fallback_to_nav2 = self.get_parameter(
            'direct_nav_fallback_to_nav2'
        ).value
        self.direct_nav_min_clearance = self.get_parameter(
            'direct_nav_min_clearance'
        ).value

        self.map_manager = MapManager(w=self.w, obstacle_buffer=self.obstacle_buffer)
        self.sampler = ProgressiveSampler(
            w=self.w,
            delta=self.delta,
            sweep_direction=self.sweep_direction,
            obstacle_buffer=self.obstacle_buffer,
            frontier_min_cells=self.frontier_min_cells,
        )
        self.rcg = RCG(w=self.w)
        self.selector = WaypointSelector(w=self.w)
        self.navigator = None  # initialised in run_coverage

        sensor_configs = load_sensor_configs_from_params(self)
        self.sensor_manager = SensorManager(
            self, sensor_configs,
            grid_resolution=0.10, splash_radius=(self.w / 2.0),
        )

        # ── CV Detection Pipeline ────────────────────────────────────
        model_dir = os.path.join(
            os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )),
            'model',
        )
        detection_configs = load_detection_configs_from_params(self)
        self.detection_manager = DetectionManager(
            self, model_dir, detection_configs,
            splash_radius=(self.w / 2.0),
            grid_resolution=0.10,
        )
        # ─────────────────────────────────────────────────────────────

        self.current_node_id = None
        self.coverage_complete = False
        self.coverage_running = False
        self.initialized = False
        self.visited_poses: list = []
        self.trajectory_poses: list = []   # dense real-time trajectory
        self.frontier_samples: list = []   # latest frontier samples for viz
        self._nav_skip_set: set = set()

        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self._map_cb, 10,
            callback_group=self.callback_group,
        )
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self._odom_cb, 10,
            callback_group=self.callback_group,
        )

        self.nodes_pub = self.create_publisher(MarkerArray, '/hazmap/rcg_nodes', 10)
        self.edges_pub = self.create_publisher(MarkerArray, '/hazmap/rcg_edges', 10)
        self.goal_pub = self.create_publisher(Marker, '/hazmap/current_goal', 10)
        self.path_pub = self.create_publisher(Path, '/hazmap/coverage_path', 10)
        self.traj_pub = self.create_publisher(Path, '/hazmap/robot_trajectory', 10)
        self.laps_pub = self.create_publisher(MarkerArray, '/hazmap/laps', 10)
        self.frontier_pub = self.create_publisher(
            MarkerArray, '/hazmap/frontier_points', 10
        )

        self.create_service(
            Trigger, '/hazmap/start_coverage',
            self._start_cb, callback_group=self.callback_group,
        )
        self.create_service(
            Trigger, '/hazmap/stop_coverage',
            self._stop_cb, callback_group=self.callback_group,
        )

        self.viz_timer = self.create_timer(
            1.0 / self.viz_rate, self._publish_viz,
            callback_group=self.callback_group,
        )

        self.traj_timer = self.create_timer(
            0.25, self._record_trajectory,
            callback_group=self.callback_group,
        )

        self.coverage_thread = None

        self.get_logger().info(
            f'HazMap node ready: w={self.w}m, δ={self.delta}, '
            f'sweep={self.sweep_direction}'
        )
        self.get_logger().info(
            'Hybrid nav: '
            f'enabled={self.use_hybrid_navigation}, '
            f'same_lap_only={self.direct_nav_same_lap_only}, '
            f'max_dist={self.direct_nav_max_distance:.2f}m, '
            f'min_clearance={self.direct_nav_min_clearance:.2f}m'
        )

    def _map_cb(self, msg: OccupancyGrid):
        q = msg.info.origin.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        self.map_manager.update_map(
            data=msg.data,
            width=msg.info.width,
            height=msg.info.height,
            resolution=msg.info.resolution,
            origin_x=msg.info.origin.position.x,
            origin_y=msg.info.origin.position.y,
            origin_yaw=yaw,
        )

    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        self.map_manager.update_robot_position(p.x, p.y, yaw)

    def _start_cb(self, req, resp):
        if self.coverage_running:
            resp.success = False
            resp.message = 'Coverage already running'
            return resp
        if not self.map_manager.map_received:
            resp.success = False
            resp.message = 'No map received yet'
            return resp
        self.coverage_thread = threading.Thread(
            target=self.run_coverage, daemon=True
        )
        self.coverage_thread.start()
        resp.success = True
        resp.message = 'Coverage started'
        return resp

    def _stop_cb(self, req, resp):
        self.coverage_running = False
        if self.navigator:
            self.navigator.cancel_navigation()
        resp.success = True
        resp.message = 'Coverage stopped'
        return resp

    def run_coverage(self):
        """Complete HazMap coverage algorithm — runs in a dedicated thread."""
        self.coverage_running = True
        self.get_logger().info('═══ HAZMAP COVERAGE STARTING ═══')

        self.get_logger().info('Waiting for /map …')
        t0 = time.time()
        while not self.map_manager.map_received:
            if time.time() - t0 > 30.0:
                self.get_logger().error('Timeout waiting for /map!')
                self.coverage_running = False
                return
            time.sleep(0.5)

        self.get_logger().info('Map received — waiting for stabilisation …')
        time.sleep(3.0)

        self.navigator = Navigator(
            self,
            pose_provider=self._get_robot_pose,
            safety_check=self._direct_safety_check,
            direct_enabled=self.use_hybrid_navigation,
            direct_linear_speed=self.direct_nav_linear_speed,
            direct_angular_speed=self.direct_nav_angular_speed,
            direct_xy_tolerance=self.direct_nav_xy_tolerance,
            direct_yaw_tolerance=self.direct_nav_yaw_tolerance,
            direct_timeout=self.direct_nav_timeout,
            direct_fallback_to_nav2=self.direct_nav_fallback_to_nav2,
        )
        if not self.navigator.is_server_ready():
            self.get_logger().error('Nav2 action server not ready!')
            self.coverage_running = False
            return

        robot_x, robot_y = self.map_manager.get_robot_position()
        self.sampler.set_start(robot_x, robot_y)
        self.get_logger().info(f'Robot start: ({robot_x:.2f}, {robot_y:.2f})')

        self.get_logger().info('Initial 360° rotation for SLAM discovery …')
        self.navigator.rotate_360(angular_speed=0.5)
        time.sleep(1.0)  # let map update after rotation

        new_samples = self.sampler.progressive_sample(self.map_manager)
        self.frontier_samples = list(new_samples)  # store for viz
        if not new_samples:
            self.get_logger().error('No frontier samples at start!')
            self.coverage_running = False
            return

        new_ids = self.rcg.expand(new_samples, self.map_manager)
        self.rcg.prune(self.map_manager, new_ids, protected_ids=set())

        self.current_node_id = self.rcg.get_nearest_node(robot_x, robot_y)
        if self.current_node_id is None:
            self.get_logger().error('No starting node found!')
            self.coverage_running = False
            return

        self.initialized = True
        self._add_pose(robot_x, robot_y)
        self.get_logger().info(
            f'Initial RCG: {self.rcg.get_node_count()} nodes, '
            f'{self.rcg.get_edge_count()} edges, '
            f'start node {self.current_node_id}'
        )

        outer_iter = 0
        reposition_retries = 0          # prevent infinite outer-loop retries
        MAX_REPOSITION_RETRIES = 3      # give up after this many

        while self.coverage_running:
            outer_iter += 1
            open_n = self.rcg.get_open_count()
            self.get_logger().info(
                f'═══ OUTER {outer_iter} ═══  '
                f'nodes={self.rcg.get_node_count()}, OPEN={open_n}'
            )

            step = 0
            consecutive_fails = 0   # stall detector

            while self.coverage_running:
                step += 1

                n_auto = self.rcg.auto_close_covered_nodes(self.map_manager)
                if n_auto > 0 and self.debug:
                    self.get_logger().info(
                        f'  Auto-closed {n_auto} already-covered node(s)'
                    )
                if self.rcg.get_open_count() == 0:
                    self.get_logger().info(
                        'All nodes auto-closed (area already covered).'
                    )
                    break

                next_id, is_dead_end = self.selector.select_next(
                    self.rcg, self.current_node_id,
                    skip_ids=self._nav_skip_set,
                )

                if is_dead_end:
                    if (self.current_node_id in self.rcg.nodes
                            and self.rcg.nodes[self.current_node_id].state
                            == NodeState.OPEN):
                        self.rcg.close_node(self.current_node_id)

                    cur = self.rcg.nodes.get(self.current_node_id)
                    if cur is None:
                        self.get_logger().warn('Current node gone!')
                        break

                    target_open = self.rcg.get_nearest_open_node_by_path(
                        self.current_node_id,
                        exclude=self._nav_skip_set,
                    )
                    if target_open is None:
                        self.get_logger().info(
                            'Dead-end, no reachable OPEN node → inner exhausted.'
                        )
                        break  # exit inner loop

                    self.get_logger().info(
                        f'  Dead-end at {self.current_node_id} L{cur.base_lap_index}, '
                        f'escaping via graph to OPEN node {target_open.id} '
                        f'L{target_open.base_lap_index}'
                    )
                    self._navigate_graph_path(
                        self.current_node_id, target_open.id
                    )
                    consecutive_fails = 0
                    continue

                self.selector.update_node_state(
                    self.rcg, self.current_node_id, next_id
                )

                if next_id is None or next_id not in self.rcg.nodes:
                    self.get_logger().warn(f'Node {next_id} missing!')
                    break

                target = self.rcg.nodes[next_id]
                cur_node = self.rcg.nodes.get(self.current_node_id)
                same_lap = (cur_node and cur_node.lap_index == target.lap_index)
                self.get_logger().info(
                    f'  Step {step}: {self.current_node_id}→{next_id} '
                    f'L{target.base_lap_index} ({target.x:.2f},{target.y:.2f}) '
                    f'[{"same" if same_lap else "cross"}-lap, direct]'
                )
                self._publish_goal(target.x, target.y)

                success = self.navigator.go_to(
                    target.x, target.y,
                    prefer_direct=True,
                )

                if not success:
                    consecutive_fails += 1

                    self._nav_skip_set.add(next_id)

                    if consecutive_fails >= 5:
                        self.get_logger().error(
                            f'Stalled: {consecutive_fails} consecutive '
                            f'nav failures — breaking inner loop.'
                        )
                        break

                    self.get_logger().warn(
                        f'  Direct nav to {next_id} failed '
                        f'(fail #{consecutive_fails}) — backing up + rotating …'
                    )
                    self.navigator.backup(
                        distance=0.30, speed=0.10, rotate_angle=0.5
                    )
                    continue

                consecutive_fails = 0   # reset stall counter
                self._nav_skip_set.clear()  # robot moved → retry skipped
                prev_node = self.rcg.nodes.get(self.current_node_id)
                self.current_node_id = next_id
                arrived = self.rcg.nodes[self.current_node_id]
                self._add_pose(arrived.x, arrived.y)

                if prev_node is not None:
                    self.map_manager.mark_edge_covered(
                        prev_node.x, prev_node.y,
                        arrived.x, arrived.y,
                    )

                rx, ry = self.map_manager.get_robot_position()
                prox_radius = self.w / 2.0
                n_prox = self.rcg.close_nodes_near_position(
                    rx, ry, prox_radius, map_manager=self.map_manager
                )
                if n_prox and self.debug:
                    self.get_logger().info(
                        f'  Proximity-closed {n_prox} nodes near robot'
                    )

                self.rcg.update_retreat_nodes_near_robot(rx, ry)

                open_count = self.rcg.get_open_count()
                self.get_logger().info(
                    f'  Arrived. OPEN={open_count}, '
                    f'retreat={len(self.rcg.retreat_nodes)}'
                )

                if open_count == 0:
                    self.get_logger().info(
                        'All current RCG nodes visited.'
                    )
                    break  # exit inner loop

                time.sleep(0.05)  # yield for callbacks

            if not self.coverage_running:
                break

            self.get_logger().info(
                'Inner loop done — 360° rotation for SLAM discovery …'
            )
            self.navigator.rotate_360(angular_speed=0.5)
            time.sleep(1.0)  # let map update after rotation

            self.get_logger().info('Checking for new sampling front …')
            new_samples = self.sampler.progressive_sample(self.map_manager)
            self.frontier_samples = list(new_samples)  # store for viz

            if not new_samples:
                remaining_open = self.rcg.get_open_count()
                if remaining_open > 0:
                    self.get_logger().warn(
                        f'No new sampling front, but {remaining_open} OPEN node(s) remain.'
                    )
                    nearest_open = self.rcg.get_nearest_open_node_by_path(
                        self.current_node_id,
                        exclude=self._nav_skip_set,
                    )
                    if nearest_open is not None:
                        reposition_retries += 1
                        if reposition_retries > MAX_REPOSITION_RETRIES:
                            self.get_logger().warn(
                                f'Gave up repositioning after '
                                f'{MAX_REPOSITION_RETRIES} retries — '
                                f'declaring coverage complete.'
                            )
                            break  # → coverage complete

                        self.get_logger().info(
                            f'Repositioning to remaining OPEN node '
                            f'{nearest_open.id}  (attempt {reposition_retries})'
                        )
                        self._navigate_graph_path(
                            self.current_node_id, nearest_open.id
                        )
                        rx, ry = self.map_manager.get_robot_position()
                        ok = (
                            math.hypot(
                                rx - nearest_open.x, ry - nearest_open.y
                            ) < self.w
                        )
                        if ok:
                            reposition_retries = 0  # reset on success
                            self.current_node_id = nearest_open.id
                            self._add_pose(nearest_open.x, nearest_open.y)
                            continue
                        self.get_logger().warn(
                            f'Failed to reach OPEN node {nearest_open.id}; '
                            'adding to skip-set.'
                        )
                        self._nav_skip_set.add(nearest_open.id)
                        continue

                self.get_logger().info(
                    '╔══════════════════════════════════╗\n'
                    '║      COVERAGE COMPLETE!          ║\n'
                    '║  No new sampling front found.    ║\n'
                    '╚══════════════════════════════════╝'
                )
                self.coverage_complete = True
                self.sensor_manager.save_results(
                    self.get_logger(),
                    detection_manager=self.detection_manager,
                )
                break

            self.get_logger().info(
                f'Found {len(new_samples)} new frontier samples — '
                f'expanding RCG …'
            )
            new_ids = self.rcg.expand(new_samples, self.map_manager)
            self.rcg.prune(
                self.map_manager, new_ids,
                protected_ids={self.current_node_id},
            )
            self.get_logger().info(
                f'RCG now {self.rcg.get_node_count()} nodes, '
                f'{self.rcg.get_open_count()} OPEN'
            )

            rx, ry = self.map_manager.get_robot_position()
            nearest = self.rcg.get_nearest_open_node(rx, ry)
            if nearest is None:
                self.get_logger().info('No OPEN nodes after expansion. Done.')
                self.coverage_complete = True
                self.sensor_manager.save_results(
                    self.get_logger(),
                    detection_manager=self.detection_manager,
                )
                break

            if nearest.id != self.current_node_id:
                self.get_logger().info(
                    f'Moving to nearest new node {nearest.id}'
                )
                self._navigate_graph_path(
                    self.current_node_id, nearest.id
                )

            self.current_node_id = nearest.id

        self.get_logger().info(
            f'HazMap finished.  Outer iterations: {outer_iter}'
        )
        self.sensor_manager.save_results(
            self.get_logger(),
            detection_manager=self.detection_manager,
        )
        self.coverage_running = False

    def _navigate_graph_path(self, from_id: int, to_id: int):
        path = self.rcg.find_graph_path(from_id, to_id)
        self.get_logger().info(
            f'  Graph-path {from_id}→{to_id}: {len(path)} hops'
        )

        for pnid in path:
            if not self.coverage_running:
                break
            if pnid not in self.rcg.nodes:
                continue
            pnode = self.rcg.nodes[pnid]
            success = self.navigator.go_to(
                pnode.x, pnode.y, prefer_direct=True,
            )
            if success:
                prev_gp = self.rcg.nodes.get(self.current_node_id)
                if prev_gp is not None:
                    self.map_manager.mark_edge_covered(
                        prev_gp.x, prev_gp.y, pnode.x, pnode.y,
                    )
                if pnode.state == NodeState.OPEN:
                    self.rcg.close_node(pnid)
                self.current_node_id = pnid
                self._add_pose(pnode.x, pnode.y)
                rx, ry = self.map_manager.get_robot_position()
                self.rcg.close_nodes_near_position(
                    rx, ry, self.w / 2.0, map_manager=self.map_manager
                )
                self.rcg.update_retreat_nodes_near_robot(rx, ry)
            else:
                self.get_logger().warn(
                    f'  Graph-path: failed to reach {pnid}, skipping'
                )
                self._nav_skip_set.add(pnid)

    def _get_robot_pose(self):
        return (
            self.map_manager.robot_x,
            self.map_manager.robot_y,
            self.map_manager.robot_yaw,
        )

    def _find_best_retreat(self, cur_node) -> 'RCGNode | None':
        """Find best retreat node, preferring same or adjacent laps."""
        from .utils import euclidean_distance as _ed

        best = None
        best_dist = float('inf')
        best_priority = 99

        for nid in self.rcg.retreat_nodes:
            if nid not in self.rcg.nodes:
                continue
            rn = self.rcg.nodes[nid]
            if rn.state != NodeState.OPEN:
                continue

            lap_diff = abs(rn.lap_index - cur_node.lap_index)
            if lap_diff == 0:
                priority = 0
            elif lap_diff == 1:
                priority = 1
            elif lap_diff == 2:
                priority = 2
            else:
                priority = 3

            dist = _ed(cur_node.x, cur_node.y, rn.x, rn.y)

            if (priority < best_priority
                    or (priority == best_priority and dist < best_dist)):
                best = rn
                best_dist = dist
                best_priority = priority

        return best

    def _direct_safety_check(
        self, rx: float, ry: float, tx: float, ty: float
    ) -> bool:
        """Reactive collision check called by Navigator during direct motion."""
        from .utils import world_to_grid as _w2g
        if self.map_manager.occupancy_grid is None:
            return False
        r0, c0 = _w2g(
            rx, ry,
            self.map_manager.origin_x, self.map_manager.origin_y,
            self.map_manager.resolution,
        )
        r1, c1 = _w2g(
            tx, ty,
            self.map_manager.origin_x, self.map_manager.origin_y,
            self.map_manager.resolution,
        )
        return is_collision_free(
            r0, c0, r1, c1, self.map_manager.occupancy_grid
        )

    def _should_prefer_direct_transition(self, from_id: int, to_id: int) -> bool:
        """Use direct local control for short local edges in normal sweep."""
        if not self.use_hybrid_navigation:
            return False
        if from_id not in self.rcg.nodes or to_id not in self.rcg.nodes:
            return False

        src = self.rcg.nodes[from_id]
        dst = self.rcg.nodes[to_id]
        dist = math.hypot(dst.x - src.x, dst.y - src.y)
        if dist > self.direct_nav_max_distance:
            return False
        if self.direct_nav_same_lap_only and src.lap_index != dst.lap_index:
            return False
        if self.map_manager.occupancy_grid is None:
            return False
        if not is_collision_free(
            src.grid_row,
            src.grid_col,
            dst.grid_row,
            dst.grid_col,
            self.map_manager.occupancy_grid,
        ):
            return False
        src_clear = self.map_manager.distance_to_nearest_obstacle(
            src.grid_row, src.grid_col
        )
        dst_clear = self.map_manager.distance_to_nearest_obstacle(
            dst.grid_row, dst.grid_col
        )
        if min(src_clear, dst_clear) < self.direct_nav_min_clearance:
            return False
        return True

    def _add_pose(self, x: float, y: float):
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.w = 1.0
        self.visited_poses.append(pose)

    def _publish_goal(self, x: float, y: float):
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'current_goal'
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = 0.2
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.2
        m.color.r = 1.0
        m.color.b = 1.0
        m.color.a = 0.9
        self.goal_pub.publish(m)

    def _publish_viz(self):
        if not self.initialized:
            return
        removed = self.rcg.revalidate_edges(self.map_manager)
        if removed > 0:
            if self.debug:
                self.get_logger().info(
                    f'Edge revalidation: removed {removed} stale edge(s)'
                )
            added = self.rcg.repair_connectivity(
                self.map_manager, anchor_id=self.current_node_id
            )
            if added > 0 and self.debug:
                self.get_logger().info(
                    f'Repair connectivity: added {added} bridge edge(s)'
                )
        now = self.get_clock().now().to_msg()
        self._pub_nodes(now)
        self._pub_edges(now)
        self._pub_laps(now)
        self._pub_path(now)
        self._pub_frontier_points(now)

    def _pub_nodes(self, stamp):
        ma = MarkerArray()
        d = Marker()
        d.header.frame_id = 'map'
        d.header.stamp = stamp
        d.ns = 'rcg_nodes'
        d.action = Marker.DELETEALL
        ma.markers.append(d)
        dl = Marker()
        dl.header.frame_id = 'map'
        dl.header.stamp = stamp
        dl.ns = 'rcg_labels'
        dl.action = Marker.DELETEALL
        ma.markers.append(dl)

        for node in self.rcg.nodes.values():
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = stamp
            m.ns = 'rcg_nodes'
            m.id = node.id + 1
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = node.x
            m.pose.position.y = node.y
            m.pose.position.z = 0.1
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.08

            if node.id == self.current_node_id:
                m.color.r, m.color.g, m.color.b = 1.0, 1.0, 0.0
            elif node.id in self.rcg.retreat_nodes:
                m.color.r, m.color.g, m.color.b = 0.0, 0.5, 1.0
            elif node.state == NodeState.OPEN:
                m.color.r, m.color.g, m.color.b = 0.0, 1.0, 0.0
            else:
                m.color.r, m.color.g, m.color.b = 1.0, 0.0, 0.0
            m.color.a = 0.9
            ma.markers.append(m)

            t = Marker()
            t.header.frame_id = 'map'
            t.header.stamp = stamp
            t.ns = 'rcg_labels'
            t.id = node.id + 1
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position.x = node.x
            t.pose.position.y = node.y
            t.pose.position.z = 0.22
            t.pose.orientation.w = 1.0
            t.scale.z = 0.05
            t.color.r = t.color.g = t.color.b = 1.0
            t.color.a = 0.7
            t.text = f'L{node.base_lap_index}'
            ma.markers.append(t)

        self.nodes_pub.publish(ma)

    def _pub_frontier_points(self, stamp):
        """Publish small red dots at every raw frontier sample position."""
        ma = MarkerArray()
        d = Marker()
        d.header.frame_id = 'map'
        d.header.stamp = stamp
        d.ns = 'frontier_pts'
        d.action = Marker.DELETEALL
        ma.markers.append(d)

        for i, fs in enumerate(self.frontier_samples):
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = stamp
            m.ns = 'frontier_pts'
            m.id = i + 1
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = fs.x
            m.pose.position.y = fs.y
            m.pose.position.z = 0.05
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.04  # small dot
            m.color.r = 1.0
            m.color.g = 0.0
            m.color.b = 0.0
            m.color.a = 0.95
            ma.markers.append(m)

        self.frontier_pub.publish(ma)

    def _pub_edges(self, stamp):
        ma = MarkerArray()
        d = Marker()
        d.header.frame_id = 'map'
        d.header.stamp = stamp
        d.ns = 'rcg_edges'
        d.action = Marker.DELETEALL
        ma.markers.append(d)

        eid = 1
        drawn = set()
        for node in self.rcg.nodes.values():
            for nid in node.neighbors:
                key = (min(node.id, nid), max(node.id, nid))
                if key in drawn:
                    continue
                drawn.add(key)
                nb = self.rcg.nodes.get(nid)
                if nb is None:
                    continue
                m = Marker()
                m.header.frame_id = 'map'
                m.header.stamp = stamp
                m.ns = 'rcg_edges'
                m.id = eid
                m.type = Marker.LINE_STRIP
                m.action = Marker.ADD
                p1 = Point(x=node.x, y=node.y, z=0.05)
                p2 = Point(x=nb.x, y=nb.y, z=0.05)
                m.points = [p1, p2]
                m.scale.x = 0.015
                if node.lap_index == nb.lap_index:
                    m.color.r, m.color.g, m.color.b = 1.0, 1.0, 0.0
                else:
                    m.color.r, m.color.g, m.color.b = 0.0, 1.0, 1.0
                m.color.a = 0.6
                ma.markers.append(m)
                eid += 1
        self.edges_pub.publish(ma)

    def _pub_laps(self, stamp):
        ma = MarkerArray()
        d = Marker()
        d.header.frame_id = 'map'
        d.header.stamp = stamp
        d.ns = 'laps'
        d.action = Marker.DELETEALL
        ma.markers.append(d)
        if not self.rcg.nodes:
            self.laps_pub.publish(ma)
            return

        import colorsys
        is_x = (self.sweep_direction == "x")

        laps: dict = {}
        for n in self.rcg.nodes.values():
            lap_id = n.base_lap_index
            if lap_id not in laps:
                if is_x:
                    laps[lap_id] = {'pos': n.x, 'mn': n.y, 'mx': n.y}
                else:
                    laps[lap_id] = {'pos': n.y, 'mn': n.x, 'mx': n.x}
            else:
                if is_x:
                    laps[lap_id]['mn'] = min(laps[lap_id]['mn'], n.y)
                    laps[lap_id]['mx'] = max(laps[lap_id]['mx'], n.y)
                else:
                    laps[lap_id]['mn'] = min(laps[lap_id]['mn'], n.x)
                    laps[lap_id]['mx'] = max(laps[lap_id]['mx'], n.x)

        lid = 1
        for li, info in laps.items():
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = stamp
            m.ns = 'laps'
            m.id = lid
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            if is_x:
                m.points = [
                    Point(x=info['pos'], y=info['mn'] - 0.15, z=0.02),
                    Point(x=info['pos'], y=info['mx'] + 0.15, z=0.02),
                ]
            else:
                m.points = [
                    Point(x=info['mn'] - 0.15, y=info['pos'], z=0.02),
                    Point(x=info['mx'] + 0.15, y=info['pos'], z=0.02),
                ]
            m.scale.x = 0.01
            r, g, b = colorsys.hsv_to_rgb((li * 0.15) % 1.0, 0.7, 0.9)
            m.color.r, m.color.g, m.color.b = float(r), float(g), float(b)
            m.color.a = 0.35
            ma.markers.append(m)
            lid += 1
        self.laps_pub.publish(ma)

    def _record_trajectory(self):
        """Sample robot position, append to trajectory, and record sensors (4 Hz)."""
        if not self.coverage_running:
            return
        rx, ry = self.map_manager.robot_x, self.map_manager.robot_y
        ryaw = self.map_manager.robot_yaw
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = rx
        pose.pose.position.y = ry
        pose.pose.orientation.z = math.sin(ryaw / 2.0)
        pose.pose.orientation.w = math.cos(ryaw / 2.0)
        self.trajectory_poses.append(pose)

        self.sensor_manager.sample_and_publish()

    def _pub_path(self, stamp):
        if self.visited_poses:
            p = Path()
            p.header.frame_id = 'map'
            p.header.stamp = stamp
            p.poses = self.visited_poses
            self.path_pub.publish(p)

        if self.trajectory_poses:
            t = Path()
            t.header.frame_id = 'map'
            t.header.stamp = stamp
            t.poses = self.trajectory_poses
            self.traj_pub.publish(t)


def main(args=None):
    rclpy.init(args=args)
    node = HazMapNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.sensor_manager.save_results(
            node.get_logger(),
            detection_manager=node.detection_manager,
        )
        node.coverage_running = False
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

