import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PoseStamped
from std_srvs.srv import Trigger
from nav2_msgs.action import NavigateToPose
import math
import os
import time
import threading

from .occupancy_grid_manager import OccupancyGridManager
from .progressive_sampling import ProgressiveSampler
from .rcg import RCG, NodeState
from .goal_selection import GoalSelector
from .tsp_solver import TSPSolver, TSPPlan
from .navigator import Navigator
from .spiral_stc import SpiralSTCPlanner
from .boustrophedon import BoustrophedonPlanner
from .next_best_view import NextBestViewSelector


class HazMapNode(Node):
    """Main ROS2 node for HazMap coverage path planning."""

    def __init__(self):
        super().__init__('hazmap_node')

        self.callback_group = ReentrantCallbackGroup()

        self.declare_parameter('w', 0.50)
        self.declare_parameter('rc', 0.30)                        # coverage radius
        self.declare_parameter('rd', 3.0)                         # detection radius
        self.declare_parameter('delta', 1)                        # kept for param-file compat
        self.declare_parameter('sweep_direction', 'x')
        self.declare_parameter('obstacle_buffer', 0.30)           # kept for param-file compat
        self.declare_parameter('visualization_rate', 2.0)
        self.declare_parameter('frontier_min_cells', 5)           # kept for param-file compat
        self.declare_parameter('debug_logging', False)
        self.declare_parameter('use_hybrid_navigation', False)
        self.declare_parameter('direct_nav_same_lap_only', True)
        self.declare_parameter('direct_nav_max_distance', 0.80)
        self.declare_parameter('direct_nav_linear_speed', 0.12)
        self.declare_parameter('direct_nav_angular_speed', 0.80)
        self.declare_parameter('direct_nav_xy_tolerance', 0.08)
        self.declare_parameter('direct_nav_yaw_tolerance', 0.20)
        self.declare_parameter('direct_nav_timeout', 20.0)
        self.declare_parameter('direct_nav_fallback_to_nav2', True)
        self.declare_parameter('direct_nav_min_clearance', 0.20)
        self.declare_parameter('known_map_mode', False)
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('auto_start_coverage', False)
        self.declare_parameter('coverage_mode', 'cstar')
        self.declare_parameter('stc_cell_size', 0.5)
        self.declare_parameter('lawnmower_spacing', 0.70)
        self.declare_parameter('lawnmower_boundary_margin', 0.70)
        self.declare_parameter('lawnmower_waypoint_retries', 3)
        self.declare_parameter('lawnmower_max_consecutive_failures', 6)
        self.declare_parameter('lawnmower_min_completion_ratio', 0.90)
        self.declare_parameter('coverage_target_percent', 95.0)
        self.declare_parameter('coverage_refine_max_goals', 120)
        self.declare_parameter('coverage_refine_max_nav_failures', 12)
        self.declare_parameter('coverage_refine_total_failures_limit', 20)
        self.declare_parameter('coverage_refine_timeout_sec', 300.0)
        self.declare_parameter('coverage_refine_search_range', 100.0)
        self.declare_parameter('coverage_refine_no_gain_patience', 10)
        self.declare_parameter('coverage_refine_min_gain_percent', 0.20)
        # ── Next-Best-View (coverage_mode: 'nbv') ────────────────────────
        self.declare_parameter('sensor_range', 0.0)          # 0 → fall back to rd
        self.declare_parameter('sensor_fov_deg', 360.0)
        self.declare_parameter('nbv_n_rays', 72)
        self.declare_parameter('nbv_q_min', 0.4)             # quality → "observed"
        self.declare_parameter('nbv_target_percent', 90.0)   # observed-coverage goal
        self.declare_parameter('nbv_cost_weight', 1.0)       # travel-cost exponent
        self.declare_parameter('nbv_max_candidates', 40)
        self.declare_parameter('nbv_min_gain', 1.0)          # quality units to bother
        self.declare_parameter('nbv_no_gain_patience', 8)

        self.w = self.get_parameter('w').value
        self.rc = self.get_parameter('rc').value
        self.rd = self.get_parameter('rd').value
        self.sweep_direction = self.get_parameter('sweep_direction').value
        self.viz_rate = self.get_parameter('visualization_rate').value
        self.debug = self.get_parameter('debug_logging').value
        self.use_hybrid_navigation = self.get_parameter('use_hybrid_navigation').value
        self.direct_nav_same_lap_only = self.get_parameter('direct_nav_same_lap_only').value
        self.direct_nav_max_distance = self.get_parameter('direct_nav_max_distance').value
        self.direct_nav_linear_speed = self.get_parameter('direct_nav_linear_speed').value
        self.direct_nav_angular_speed = self.get_parameter('direct_nav_angular_speed').value
        self.direct_nav_xy_tolerance = self.get_parameter('direct_nav_xy_tolerance').value
        self.direct_nav_yaw_tolerance = self.get_parameter('direct_nav_yaw_tolerance').value
        self.direct_nav_timeout = self.get_parameter('direct_nav_timeout').value
        self.direct_nav_fallback_to_nav2 = self.get_parameter('direct_nav_fallback_to_nav2').value
        self.direct_nav_min_clearance = self.get_parameter('direct_nav_min_clearance').value
        self.known_map_mode = self.get_parameter('known_map_mode').value
        self.map_topic = self.get_parameter('map_topic').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.scan_topic = self.get_parameter('scan_topic').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.auto_start_coverage = self.get_parameter('auto_start_coverage').value
        self.coverage_mode = self.get_parameter('coverage_mode').value
        self.stc_cell_size = self.get_parameter('stc_cell_size').value
        self.lawnmower_spacing = self.get_parameter('lawnmower_spacing').value
        self.lawnmower_boundary_margin = self.get_parameter('lawnmower_boundary_margin').value
        self.lawnmower_waypoint_retries = self.get_parameter('lawnmower_waypoint_retries').value
        self.lawnmower_max_consecutive_failures = self.get_parameter(
            'lawnmower_max_consecutive_failures'
        ).value
        self.lawnmower_min_completion_ratio = self.get_parameter(
            'lawnmower_min_completion_ratio'
        ).value
        self.coverage_target_percent = self.get_parameter(
            'coverage_target_percent'
        ).value
        self.coverage_refine_max_goals = self.get_parameter(
            'coverage_refine_max_goals'
        ).value
        self.coverage_refine_max_nav_failures = self.get_parameter(
            'coverage_refine_max_nav_failures'
        ).value
        self.coverage_refine_total_failures_limit = self.get_parameter(
            'coverage_refine_total_failures_limit'
        ).value
        self.coverage_refine_timeout_sec = self.get_parameter(
            'coverage_refine_timeout_sec'
        ).value
        self.coverage_refine_search_range = self.get_parameter(
            'coverage_refine_search_range'
        ).value
        self.coverage_refine_no_gain_patience = self.get_parameter(
            'coverage_refine_no_gain_patience'
        ).value
        self.coverage_refine_min_gain_percent = self.get_parameter(
            'coverage_refine_min_gain_percent'
        ).value
        _sensor_range = self.get_parameter('sensor_range').value
        self.sensor_range = _sensor_range if _sensor_range > 0.0 else self.rd
        self.sensor_fov_deg = self.get_parameter('sensor_fov_deg').value
        self.nbv_n_rays = self.get_parameter('nbv_n_rays').value
        self.nbv_q_min = self.get_parameter('nbv_q_min').value
        self.nbv_target_percent = self.get_parameter('nbv_target_percent').value
        self.nbv_cost_weight = self.get_parameter('nbv_cost_weight').value
        self.nbv_max_candidates = self.get_parameter('nbv_max_candidates').value
        self.nbv_min_gain = self.get_parameter('nbv_min_gain').value
        self.nbv_no_gain_patience = self.get_parameter('nbv_no_gain_patience').value

        # ── C* core algorithm objects ────────────────────────────────
        self.ogm = OccupancyGridManager(free_threshold=50)
        self._robot_x = 0.0
        self._robot_y = 0.0
        self._robot_yaw = 0.0
        self._odom_received = False

        # sweep_dir: (0,1) = laps along Y (hazmap "x"), (1,0) = laps along X ("y")
        sweep_dir = (0.0, 1.0) if self.sweep_direction == 'x' else (1.0, 0.0)
        self.sampler = ProgressiveSampler(
            self.w,
            self.rd,
            sweep_dir,
            self.ogm,
            known_map_mode=self.known_map_mode,
        )
        self.rcg = RCG(self.w, self.ogm)
        self.goal_selector = GoalSelector(self.rcg, ogm=self.ogm, rc=self.rc)
        self.tsp_solver = TSPSolver(self.rcg)
        self.stc_planner = SpiralSTCPlanner(self.ogm, cell_size_m=self.stc_cell_size)
        self.boustro_planner = BoustrophedonPlanner(
            self.ogm,
            lap_spacing_m=self.lawnmower_spacing,
            boundary_margin_m=self.lawnmower_boundary_margin,
        )
        self.nbv_selector = NextBestViewSelector(
            self.rcg,
            self.ogm,
            sensor_range=self.sensor_range,
            fov_deg=self.sensor_fov_deg,
            n_rays=int(self.nbv_n_rays),
            q_min=self.nbv_q_min,
            cost_weight=self.nbv_cost_weight,
            max_candidates=int(self.nbv_max_candidates),
        )
        self.navigator = None  # initialised in run_coverage
        self._coverage_anchor = None

        # ── Visit log ──────────────────────────────────────────────────
        # Records every goal the algorithm chose: timestamp, step, node_id,
        # target world coords, "source" (which decision branch chose it),
        # nav result, coverage % at arrival. Saved as CSV on shutdown.
        self._visit_log: list = []
        self._visit_step = 0

        self.current_node_id = None
        self.coverage_complete = False
        self.coverage_running = False
        self.initialized = False
        self.visited_poses: list = []
        self.trajectory_poses: list = []
        self.frontier_samples: list = []

        map_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.map_sub = self.create_subscription(
            OccupancyGrid, self.map_topic, self._map_cb, map_qos,
            callback_group=self.callback_group,
        )
        self.odom_sub = self.create_subscription(
            Odometry, self.odom_topic, self._odom_cb, 10,
            callback_group=self.callback_group,
        )

        self.nodes_pub = self.create_publisher(MarkerArray, 'hazmap/rcg_nodes', 10)
        self.edges_pub = self.create_publisher(MarkerArray, 'hazmap/rcg_edges', 10)
        self.goal_pub = self.create_publisher(Marker, 'hazmap/current_goal', 10)
        self.path_pub = self.create_publisher(Path, 'hazmap/coverage_path', 10)
        self.traj_pub = self.create_publisher(Path, 'hazmap/robot_trajectory', 10)
        self.laps_pub = self.create_publisher(MarkerArray, 'hazmap/laps', 10)
        self.frontier_pub = self.create_publisher(
            MarkerArray, 'hazmap/frontier_points', 10
        )
        self.obs_quality_pub = self.create_publisher(
            OccupancyGrid, 'hazmap/observation_quality', map_qos
        )

        self.create_service(
            Trigger, 'hazmap/start_coverage',
            self._start_cb, callback_group=self.callback_group,
        )
        self.create_service(
            Trigger, 'hazmap/stop_coverage',
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
        self._auto_start_timer = None
        self._auto_start_wait_log_counter = 0
        self._autostart_nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.get_logger().info(
            f'HazMap node ready (C* core): w={self.w}m, rc={self.rc}m, '
            f'rd={self.rd}m, sweep={self.sweep_direction}, '
            f'known_map_mode={self.known_map_mode}, coverage_mode={self.coverage_mode}'
        )

        if self.auto_start_coverage:
            self.get_logger().info(
                'Auto-start enabled: coverage will start once /map is available.'
            )
            self._auto_start_timer = self.create_timer(
                1.0, self._auto_start_if_ready, callback_group=self.callback_group
            )

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------
    def _map_cb(self, msg: OccupancyGrid):
        self.ogm.update(msg)

    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        self._robot_x = p.x
        self._robot_y = p.y
        self._robot_yaw = yaw
        self._odom_received = True

    def _start_cb(self, req, resp):
        if self.coverage_running:
            resp.success = False
            resp.message = 'Coverage already running'
            return resp
        if not self.ogm.ready:
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

    def _auto_start_if_ready(self):
        if self.coverage_running or self.coverage_complete:
            if self._auto_start_timer is not None:
                self._auto_start_timer.cancel()
            return
        if not self.ogm.ready:
            return
        if not self._odom_received:
            return

        service_names = {name for name, _ in self.get_service_names_and_types()}
        have_odor = any(name.endswith('/odor_value') or name == 'odor_value' for name in service_names)
        have_wind = any(name.endswith('/wind_value') or name == 'wind_value' for name in service_names)
        have_gaden_services = have_odor and have_wind
        if not have_gaden_services:
            self._auto_start_wait_log_counter += 1
            if self._auto_start_wait_log_counter % 5 == 0:
                self.get_logger().info(
                    'Auto-start: waiting for GADEN services /odor_value and /wind_value ...'
                )
            return

        if not self._autostart_nav_client.wait_for_server(timeout_sec=0.0):
            self._auto_start_wait_log_counter += 1
            if self._auto_start_wait_log_counter % 5 == 0:
                self.get_logger().info(
                    'Auto-start: waiting for Nav2 action server navigate_to_pose ...'
                )
            return

        self.get_logger().info('Auto-start: map received, starting coverage.')
        self.coverage_thread = threading.Thread(target=self.run_coverage, daemon=True)
        self.coverage_thread.start()
        if self._auto_start_timer is not None:
            self._auto_start_timer.cancel()

    # ------------------------------------------------------------------
    # Main coverage loop
    # ------------------------------------------------------------------
    def run_coverage(self):
        if self.coverage_mode == 'boustrophedon_known':
            self.run_coverage_boustrophedon()
            return
        if self.coverage_mode == 'spiral_stc_known':
            self.run_coverage_spiral_stc()
            return
        if self.coverage_mode == 'nbv':
            self.run_coverage_nbv()
            return

        """HazMap coverage algorithm using C* core — runs in a dedicated thread."""
        self.coverage_running = True
        self.get_logger().info('═══ HAZMAP COVERAGE STARTING (C* core) ═══')

        self.get_logger().info('Waiting for /map …')
        t0 = time.time()
        while not self.ogm.ready:
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
            scan_topic=self.scan_topic,
            cmd_vel_topic=self.cmd_vel_topic,
        )
        if not self.navigator.is_server_ready():
            self.get_logger().error('Nav2 action server not ready!')
            self.coverage_running = False
            return

        robot_x, robot_y = self._robot_x, self._robot_y
        self.get_logger().info(f'Robot start: ({robot_x:.2f}, {robot_y:.2f})')

        if not self.known_map_mode:
            self.get_logger().info('Initial 360° rotation for SLAM discovery skipped.')
        else:
            self.get_logger().info('Known-map mode: skipping initial SLAM discovery spin.')

        robot_x, robot_y = self._robot_x, self._robot_y
        new_samples = self.sampler.generate_samples(robot_x, robot_y)
        self.frontier_samples = list(new_samples)
        if not new_samples:
            self.get_logger().error('No frontier samples at start!')
            self.coverage_running = False
            return

        new_ids = self.rcg.expand(new_samples)
        self.rcg.prune(new_ids)

        self.current_node_id = self._nearest_node_id(robot_x, robot_y)
        if self.current_node_id is None:
            self.get_logger().error('No starting node found!')
            self.coverage_running = False
            return

        self.initialized = True
        self._add_pose(robot_x, robot_y)
        self._reset_coverage_anchor(robot_x, robot_y)
        self.get_logger().info(
            f'Initial RCG: {len(self.rcg.nodes)} nodes, '
            f'{len(self.rcg.edges)} edges, '
            f'start node {self.current_node_id}'
        )

        outer_iter = 0

        # Track unreachable / repeatedly-failing nodes so the goal selector
        # stops returning them. We mark them CLOSED after MAX_NODE_FAILS misses.
        node_fail_counts: dict[int, int] = {}
        MAX_NODE_FAILS = 2

        # Track frontiers our safety-net has already tried and failed,
        # so we don't loop forever on an unreachable map cell.
        failed_frontiers: list[tuple[float, float]] = []
        # Anti-revisit: a frontier that we *reach* but that yields no new
        # area is unproductive (e.g. an opening onto space we can't actually
        # sense into). Blacklist it and count consecutive unproductive visits
        # so the run terminates instead of livelocking on the same cell.
        frontier_no_gain = 0
        FRONTIER_NO_GAIN_PATIENCE = 4
        # Region-aware injector: every N successful cstar arrivals, force a
        # jump to the largest distant frontier cluster so isolated regions
        # don't get starved while the lap selector loops locally.
        region_inject_counter = 0
        REGION_INJECT_INTERVAL = 12

        # Configurable early termination: stop when coverage_target_percent
        # is reached even in unknown-map mode. Set to >100 to disable.
        early_stop_coverage = float(self.coverage_target_percent)

        # Stagnation detector: in unknown-map mode the "coverage %" has a
        # moving denominator (new free cells appear as the rover discovers
        # area), so the percent can plateau or fall even while we're
        # actually making progress. Track the absolute number of free cells
        # discovered and the absolute area covered; if neither has grown
        # meaningfully in the last STAGNATION_PATIENCE arrivals, stop.
        STAGNATION_PATIENCE = 8
        MIN_AREA_GROWTH_M2 = 0.30  # require >= 0.3 m^2 new covered area
        MIN_FREE_GROWTH_M2 = 0.50  # or >= 0.5 m^2 new free territory discovered
        stagnation_count = 0
        prev_area_covered = 0.0
        prev_total_free = 0.0

        while self.coverage_running:
            outer_iter += 1
            open_n = self.rcg.num_open
            self.get_logger().info(
                f'═══ OUTER {outer_iter} ═══  '
                f'nodes={len(self.rcg.nodes)}, OPEN={open_n}'
            )

            step = 0
            consecutive_fails = 0

            while self.coverage_running:
                step += 1

                if self.rcg.num_open == 0:
                    self.get_logger().info('All nodes covered.')
                    break

                next_id = self.goal_selector.select_goal_node(self.current_node_id)

                if next_id is None:
                    # Dead end — escape via retreat/nearest open node
                    if (self.current_node_id in self.rcg.nodes
                            and self.rcg.nodes[self.current_node_id].state
                            == NodeState.OPEN):
                        self.rcg.set_node_state(self.current_node_id, NodeState.CLOSED)

                    cur = self.rcg.nodes.get(self.current_node_id)
                    if cur is None:
                        self.get_logger().warn('Current node gone!')
                        break

                    target_id = self.goal_selector._escape_dead_end(self.current_node_id)
                    if target_id is None:
                        self.get_logger().info(
                            'Dead-end, no reachable OPEN node → inner exhausted.'
                        )
                        break

                    target_open = self.rcg.nodes.get(target_id)
                    if target_open is None:
                        break

                    self.get_logger().info(
                        f'  Dead-end at {self.current_node_id} '
                        f'L{cur.lap_index}, escaping to {target_id}'
                    )
                    visit_idx = self._log_visit(
                        'deadend_escape',
                        target_open.x, target_open.y,
                        node_id=target_id,
                    )
                    self._navigate_graph_path(self.current_node_id, target_id)
                    self._mark_visit_result(
                        visit_idx,
                        'arrived' if self.current_node_id == target_id else 'failed',
                    )
                    consecutive_fails = 0
                    continue

                self.goal_selector.update_state(self.current_node_id, next_id)

                if next_id not in self.rcg.nodes:
                    self.get_logger().warn(f'Node {next_id} missing!')
                    break

                self._detect_and_cover_holes(self.current_node_id, next_id)

                target = self.rcg.nodes[next_id]
                self.get_logger().info(
                    f'  Step {step}: {self.current_node_id}→{next_id} '
                    f'L{target.lap_index} ({target.x:.2f},{target.y:.2f})'
                )
                self._publish_goal(target.x, target.y)
                visit_idx = self._log_visit(
                    'cstar', target.x, target.y, node_id=next_id,
                )

                success = self.navigator.go_to(target.x, target.y, prefer_direct=False)
                self._mark_visit_result(
                    visit_idx, 'arrived' if success else 'nav_failed',
                )

                if not success:
                    consecutive_fails += 1
                    node_fail_counts[next_id] = node_fail_counts.get(next_id, 0) + 1

                    # If this specific node has failed too many times, it is
                    # almost certainly unreachable (e.g., goal lands in the
                    # global-costmap inflated zone, or in an unmapped pocket).
                    # Close it so select_goal_node stops returning it.
                    if node_fail_counts[next_id] >= MAX_NODE_FAILS:
                        if next_id in self.rcg.nodes:
                            self.rcg.set_node_state(next_id, NodeState.CLOSED)
                        self.get_logger().warn(
                            f'  Node {next_id} unreachable after '
                            f'{node_fail_counts[next_id]} attempts — '
                            f'marking CLOSED.'
                        )
                        consecutive_fails = 0
                        continue

                    if consecutive_fails >= 5:
                        self.get_logger().error(
                            f'Stalled: {consecutive_fails} consecutive '
                            f'nav failures — breaking inner loop.'
                        )
                        break
                    self.get_logger().warn(
                        f'  Nav to {next_id} failed '
                        f'(fail #{consecutive_fails}) — backing up …'
                    )
                    self.navigator.backup(distance=0.30, speed=0.10, rotate_angle=0.5)
                    continue

                consecutive_fails = 0
                node_fail_counts.pop(next_id, None)
                prev_node = self.rcg.nodes.get(self.current_node_id)
                self.current_node_id = next_id
                arrived = self.rcg.nodes[self.current_node_id]
                self._add_pose(arrived.x, arrived.y)

                # Mark coverage swath
                self._mark_covered_to(arrived.x, arrived.y)

                rx, ry = self._robot_x, self._robot_y
                self.rcg.close_nearby_nodes(rx, ry, self.rc)
                self.goal_selector.update_retreat_nodes(rx, ry)

                # Incremental sampling + RCG expansion (Algorithm 4 lines 3-7)
                inc_samples = self.sampler.generate_samples(rx, ry)
                # Off-grid cluster seeds: also add OPEN nodes at frontier
                # cluster centroids so every opening (narrow / oblique /
                # between-lap) has a goal node and the utility selector can
                # see them as candidates.
                inc_samples.extend(self._cluster_samples_from_grid())
                if inc_samples:
                    self.frontier_samples = list(inc_samples)
                    inc_ids = self.rcg.expand(inc_samples)
                    self.rcg.prune(inc_ids)
                    if self.debug:
                        self.get_logger().info(
                            f'  Incremental: {len(inc_samples)} samples, '
                            f'{len(inc_ids)} new nodes'
                        )

                # Region-aware injector: every N cstar arrivals, force a
                # cross-map jump to the largest distant frontier cluster, so
                # remote unexplored regions don't get starved by local lap
                # selection (the bottom-right-unknown case).
                region_inject_counter += 1
                if region_inject_counter >= REGION_INJECT_INTERVAL:
                    region_inject_counter = 0
                    self._inject_region_jump(rx, ry, failed_frontiers)

                # Early termination: declare done once enough area has been seen,
                # OR once we've stopped meaningfully discovering / covering area.
                pct, area_covered, total_free = self._coverage_stats()
                area_growth = area_covered - prev_area_covered
                free_growth = total_free - prev_total_free
                if (area_growth < MIN_AREA_GROWTH_M2
                        and free_growth < MIN_FREE_GROWTH_M2):
                    stagnation_count += 1
                else:
                    stagnation_count = 0
                prev_area_covered = area_covered
                prev_total_free = total_free

                open_count = self.rcg.num_open
                self.get_logger().info(
                    f'  Arrived. OPEN={open_count}, '
                    f'retreat={len(self.goal_selector.retreat_nodes)}, '
                    f'coverage={pct:.1f}% ({area_covered:.1f}/{total_free:.1f} m²), '
                    f'stagnant={stagnation_count}/{STAGNATION_PATIENCE}'
                )
                if pct >= early_stop_coverage:
                    self.get_logger().info(
                        f'╔══════════════════════════════════╗\n'
                        f'║   COVERAGE TARGET REACHED        ║\n'
                        f'║   {pct:.1f}% ≥ {early_stop_coverage:.1f}% — stopping.\n'
                        f'╚══════════════════════════════════╝'
                    )
                    self.coverage_complete = True
                    self.coverage_running = False
                    break

                if stagnation_count >= STAGNATION_PATIENCE:
                    self.get_logger().info(
                        f'╔══════════════════════════════════╗\n'
                        f'║   STAGNATED — STOPPING           ║\n'
                        f'║   no new area in {STAGNATION_PATIENCE} arrivals \n'
                        f'║   final coverage {pct:.1f}%\n'
                        f'╚══════════════════════════════════╝'
                    )
                    self.coverage_complete = True
                    self.coverage_running = False
                    break

                if open_count == 0:
                    self.get_logger().info('All current RCG nodes visited.')
                    break

                time.sleep(0.05)

            if not self.coverage_running:
                break

            if not self.known_map_mode:
                self.get_logger().info(
                    'Inner loop done — 360° rotation for SLAM discovery skipped.'
                )
            else:
                self.get_logger().info(
                    'Inner loop done — known-map mode, no SLAM spin needed.'
                )

            self.get_logger().info('Checking for new sampling front …')
            rx, ry = self._robot_x, self._robot_y
            new_samples = self.sampler.generate_samples(rx, ry)
            self.frontier_samples = list(new_samples)

            if not new_samples:
                remaining_open = self.rcg.num_open
                if remaining_open > 0:
                    self.get_logger().warn(
                        f'No new sampling front, but {remaining_open} OPEN node(s) remain.'
                    )

                    nearest_open = self._nearest_open_node_euclidean(rx, ry)

                    if nearest_open is not None:
                        self.get_logger().info(
                            f'Repositioning to remaining OPEN node {nearest_open.id}'
                        )
                        visit_idx = self._log_visit(
                            'outer_repos',
                            nearest_open.x, nearest_open.y,
                            node_id=nearest_open.id,
                        )
                        path = self.rcg.astar(self.current_node_id, nearest_open.id)
                        if path:
                            self._navigate_graph_path(self.current_node_id, nearest_open.id)
                            self._mark_visit_result(
                                visit_idx,
                                'arrived' if self.current_node_id == nearest_open.id else 'failed',
                            )
                        else:
                            success = self.navigator.go_to(
                                nearest_open.x, nearest_open.y, prefer_direct=False,
                            )
                            if success:
                                self.current_node_id = nearest_open.id
                                self._add_pose(nearest_open.x, nearest_open.y)
                            self._mark_visit_result(
                                visit_idx, 'arrived' if success else 'nav_failed',
                            )
                        continue

                # Safety net: before declaring done, look for unexplored
                # openings via frontier *clusters* (free-cell groups adjacent
                # to unknown). Clusters find non-lap-aligned openings the lap
                # sampler misses; excluding blacklisted frontiers makes the
                # rover sweep every reachable opening exactly once instead of
                # livelocking on the nearest unproductive cell.
                blacklist_radius = max(self.w, 0.50)
                target = self.ogm.find_nearest_frontier_cluster(
                    rx, ry,
                    min_cluster_cells=3,
                    min_distance=max(0.0, self.direct_nav_xy_tolerance),
                    exclude=failed_frontiers,
                    exclude_radius=blacklist_radius,
                )
                if target is not None:
                    tx, ty = self._clamp_goal_to_map(target[0], target[1])
                    self.get_logger().warn(
                        f'No lap samples left — repositioning to frontier '
                        f'opening ({tx:.2f},{ty:.2f}).'
                    )
                    pre_pct, pre_area, pre_free = self._coverage_stats()
                    self._publish_goal(tx, ty)
                    visit_idx = self._log_visit('safety_net', tx, ty)
                    success = self.navigator.go_to(
                        tx, ty, prefer_direct=False, timeout=60.0,
                    )
                    self._mark_visit_result(
                        visit_idx, 'arrived' if success else 'nav_failed',
                    )
                    if not success:
                        # Unreachable opening — blacklist and move on.
                        failed_frontiers.append((tx, ty))
                        frontier_no_gain += 1
                        self.get_logger().warn(
                            f'Frontier nav failed; blacklisting '
                            f'({tx:.2f},{ty:.2f}) (no_gain={frontier_no_gain}).'
                        )
                    else:
                        self._add_pose(tx, ty)
                        self._mark_covered_to(tx, ty)
                        rx, ry = self._robot_x, self._robot_y
                        # Let SLAM/coverage settle, then check whether this
                        # opening actually revealed or covered anything.
                        post_pct, post_area, post_free = self._coverage_stats()
                        gained = (
                            (post_area - pre_area) >= MIN_AREA_GROWTH_M2
                            or (post_free - pre_free) >= MIN_FREE_GROWTH_M2
                        )
                        if gained:
                            frontier_no_gain = 0
                        else:
                            # Reached, but no new area — this opening is a
                            # dead-end for sensing. Blacklist so we never
                            # return here (this is the old livelock cause).
                            failed_frontiers.append((tx, ty))
                            frontier_no_gain += 1
                            self.get_logger().warn(
                                f'Frontier ({tx:.2f},{ty:.2f}) reached but '
                                f'added no area — blacklisting '
                                f'(no_gain={frontier_no_gain}).'
                            )
                    if frontier_no_gain < FRONTIER_NO_GAIN_PATIENCE:
                        continue
                    self.get_logger().warn(
                        f'No productive frontier in '
                        f'{FRONTIER_NO_GAIN_PATIENCE} attempts — '
                        f'remaining unknown is unreachable. Completing.'
                    )

                self.get_logger().info(
                    '╔══════════════════════════════════╗\n'
                    '║      COVERAGE COMPLETE!          ║\n'
                    '║  No new sampling front found.    ║\n'
                    '╚══════════════════════════════════╝'
                )
                self.coverage_complete = True
                self._save_visit_log()
                break

            self.get_logger().info(
                f'Found {len(new_samples)} new frontier samples — expanding RCG …'
            )
            new_ids = self.rcg.expand(new_samples)
            self.rcg.prune(new_ids)
            self.get_logger().info(
                f'RCG now {len(self.rcg.nodes)} nodes, {self.rcg.num_open} OPEN'
            )

            nearest = self._nearest_open_node_euclidean(rx, ry)
            if nearest is None:
                self.get_logger().info('No OPEN nodes after expansion. Done.')
                self.coverage_complete = True
                self._save_visit_log()
                break

            if nearest.id != self.current_node_id:
                self.get_logger().info(f'Moving to nearest new node {nearest.id}')
                self._navigate_graph_path(self.current_node_id, nearest.id)

            self.current_node_id = nearest.id

        self.get_logger().info(f'HazMap finished.  Outer iterations: {outer_iter}')
        self._save_visit_log()
        self.coverage_running = False

    def run_coverage_spiral_stc(self):
        """Known-map coverage using Spiral-STC style tree traversal."""
        self.coverage_running = True
        self.coverage_complete = False
        self.initialized = True
        self.get_logger().info('═══ HAZMAP COVERAGE STARTING (Spiral-STC known map) ═══')

        self.get_logger().info('Waiting for /map …')
        t0 = time.time()
        while not self.ogm.ready:
            if time.time() - t0 > 30.0:
                self.get_logger().error('Timeout waiting for /map!')
                self.coverage_running = False
                return
            time.sleep(0.5)

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
            scan_topic=self.scan_topic,
            cmd_vel_topic=self.cmd_vel_topic,
        )
        if not self.navigator.is_server_ready():
            self.get_logger().error('Nav2 action server not ready!')
            self.coverage_running = False
            return

        rx, ry = self._robot_x, self._robot_y
        self._add_pose(rx, ry)
        self._reset_coverage_anchor(rx, ry)
        self.get_logger().info(
            f'Planning Spiral-STC route from ({rx:.2f}, {ry:.2f}), '
            f'cell_size={self.stc_cell_size:.2f}m'
        )
        waypoints = self.stc_planner.plan(rx, ry)
        if not waypoints:
            self.get_logger().warn(
                'Spiral-STC planner returned no waypoints. '
                'Falling back to C* coverage in known-map mode.'
            )
            self.coverage_running = False
            self.coverage_mode = 'cstar'
            self.run_coverage()
            return

        self.get_logger().info(f'Spiral-STC route size: {len(waypoints)} waypoints')
        # Fast path: drive the route in NavigateThroughPoses chunks; resume
        # per-waypoint from the first chunk that fails.
        resume = self._drive_route_ntp(waypoints)
        if resume:
            self.get_logger().info(
                f'Spiral-STC: NTP completed {resume}/{len(waypoints)} waypoints.'
            )
        for i, (tx, ty) in enumerate(waypoints[resume:], start=resume + 1):
            if not self.coverage_running:
                break
            rx, ry, _ = self._get_robot_pose()
            if math.hypot(tx - rx, ty - ry) < max(0.10, self.direct_nav_xy_tolerance):
                # Skip ultra-short hops that can trigger controller churn.
                self._mark_covered_to(tx, ty)
                continue
            self._publish_goal(tx, ty)
            ok = self.navigator.go_to(tx, ty, prefer_direct=True, timeout=90.0)
            if not ok:
                self.navigator.backup(distance=0.20, speed=0.08, rotate_angle=0.35)
                ok = self.navigator.go_to(tx, ty, prefer_direct=True, timeout=90.0)
            if not ok:
                self.get_logger().warn(
                    f'Spiral-STC: nav failed at {i}/{len(waypoints)} '
                    f'({tx:.2f},{ty:.2f}), continuing.'
                )
                continue
            self._add_pose(tx, ty)
            self._mark_covered_to(tx, ty)

        if not self.coverage_running:
            self.coverage_complete = False
            self.get_logger().warn('Spiral-STC coverage interrupted before completion.')
        else:
            coverage_percent = self._refine_known_map_coverage()
            self.coverage_complete = coverage_percent >= float(self.coverage_target_percent)
            if self.coverage_complete:
                self.get_logger().info(
                    'Spiral-STC coverage completed. '
                    f'area={coverage_percent:.1f}% '
                    f'(target {float(self.coverage_target_percent):.1f}%).'
                )
            else:
                self.get_logger().warn(
                    'Spiral-STC coverage ended below area target. '
                    f'area={coverage_percent:.1f}%, '
                    f'target={float(self.coverage_target_percent):.1f}%.'
                )
        self.coverage_running = False
        self._save_visit_log()

    def run_coverage_boustrophedon(self):
        """Known-map coverage using boustrophedon lawnmower strips."""
        self.coverage_running = True
        self.coverage_complete = False
        self.initialized = True
        self.get_logger().info('═══ HAZMAP COVERAGE STARTING (Boustrophedon known map) ═══')

        self.get_logger().info('Waiting for /map …')
        t0 = time.time()
        while not self.ogm.ready:
            if time.time() - t0 > 30.0:
                self.get_logger().error('Timeout waiting for /map!')
                self.coverage_running = False
                return
            time.sleep(0.5)

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
            scan_topic=self.scan_topic,
            cmd_vel_topic=self.cmd_vel_topic,
        )
        if not self.navigator.is_server_ready():
            self.get_logger().error('Nav2 action server not ready!')
            self.coverage_running = False
            return

        waypoints = self.boustro_planner.plan(sweep_axis=self.sweep_direction)
        if not waypoints:
            self.get_logger().warn(
                'Boustrophedon planner returned no waypoints. '
                'Falling back to Spiral-STC.'
            )
            self.coverage_running = False
            self.coverage_mode = 'spiral_stc_known'
            self.run_coverage_spiral_stc()
            return

        self.get_logger().info(
            f'Boustrophedon route size: {len(waypoints)} waypoints '
            f'(spacing={self.lawnmower_spacing:.2f}m)'
        )
        rx, ry, _ = self._get_robot_pose()
        self._reset_coverage_anchor(rx, ry)

        interrupted = False
        reached_waypoints = 0
        skipped_close_waypoints = 0
        failed_waypoints = 0
        consecutive_failures = 0
        # Fast path: drive in NavigateThroughPoses chunks, then resume the
        # robust per-waypoint loop from the first chunk that fails.
        resume = self._drive_route_ntp(waypoints)
        reached_waypoints += resume
        if resume:
            self.get_logger().info(
                f'Boustrophedon: NTP completed {resume}/{len(waypoints)} waypoints.'
            )
        for i, (tx, ty) in enumerate(waypoints[resume:], start=resume + 1):
            if not self.coverage_running:
                interrupted = True
                break
            tx, ty = self._clamp_goal_to_map(tx, ty)
            rx, ry, _ = self._get_robot_pose()
            if math.hypot(tx - rx, ty - ry) < max(0.10, self.direct_nav_xy_tolerance):
                self._mark_covered_to(tx, ty)
                skipped_close_waypoints += 1
                continue
            self._publish_goal(tx, ty)

            ok = False
            for attempt in range(1, int(self.lawnmower_waypoint_retries) + 1):
                ok = self.navigator.go_to(tx, ty, prefer_direct=True, timeout=90.0)
                if ok:
                    break
                self.navigator.backup(
                    distance=min(0.35, 0.15 + 0.07 * attempt),
                    speed=0.08,
                    rotate_angle=min(0.9, 0.25 + 0.2 * attempt),
                )
            if not ok:
                failed_waypoints += 1
                consecutive_failures += 1
                self.get_logger().warn(
                    f'Boustrophedon: nav failed at {i}/{len(waypoints)} '
                    f'({tx:.2f},{ty:.2f}), consecutive_failures={consecutive_failures}'
                )
                if consecutive_failures >= int(self.lawnmower_max_consecutive_failures):
                    self.get_logger().warn(
                        'Boustrophedon: too many consecutive failures, '
                        'performing escape rotate and continuing.'
                    )
                    self.navigator.rotate_360(angular_speed=0.35)
                    consecutive_failures = 0
                continue

            consecutive_failures = 0
            reached_waypoints += 1
            self._add_pose(tx, ty)
            self._mark_covered_to(tx, ty)

        completed_waypoints = reached_waypoints + skipped_close_waypoints
        completion_ratio = (
            completed_waypoints / float(len(waypoints))
            if waypoints else 0.0
        )
        meets_completion_ratio = (
            completion_ratio >= float(self.lawnmower_min_completion_ratio)
        )
        if interrupted:
            self.coverage_running = False
            reason = 'interrupted' if interrupted else 'insufficient_completion_ratio'
            self.get_logger().warn(
                'Boustrophedon coverage ended early/incomplete. '
                f'reason={reason}, completion={completion_ratio:.1%}, '
                f'required>={float(self.lawnmower_min_completion_ratio):.1%}, '
                f'completed={completed_waypoints}/{len(waypoints)}, '
                f'failed={failed_waypoints}'
            )
            self.coverage_complete = False
        else:
            if not meets_completion_ratio:
                self.get_logger().warn(
                    'Boustrophedon route completion ratio below threshold. '
                    f'completion={completion_ratio:.1%}, '
                    f'required>={float(self.lawnmower_min_completion_ratio):.1%}. '
                    'Trying coverage refinement anyway.'
                )

            coverage_percent = self._refine_known_map_coverage()
            self.coverage_running = False
            self.coverage_complete = coverage_percent >= float(self.coverage_target_percent)

            if self.coverage_complete:
                self.get_logger().info(
                    'Boustrophedon coverage completed. '
                    f'waypoints={completion_ratio:.1%}, '
                    f'area={coverage_percent:.1f}% '
                    f'(target {float(self.coverage_target_percent):.1f}%).'
                )
            else:
                self.get_logger().warn(
                    'Boustrophedon coverage ended below area target. '
                    f'waypoints={completion_ratio:.1%}, '
                    f'area={coverage_percent:.1f}%, '
                    f'target={float(self.coverage_target_percent):.1f}%.'
                )
        self._save_visit_log()

    # ------------------------------------------------------------------
    # Next-Best-View coverage (coverage_mode: 'nbv')
    # ------------------------------------------------------------------
    def run_coverage_nbv(self):
        """Information-gain coverage: repeatedly drive to the OPEN viewpoint
        that reveals the most still-unseen area, until a high observed-coverage
        target is met. Credits cells by graded line-of-sight observation
        (sensor_range) rather than by driving over them."""
        self.coverage_running = True
        self.coverage_complete = False
        self.get_logger().info('═══ HAZMAP COVERAGE STARTING (Next-Best-View) ═══')

        self.get_logger().info('Waiting for /map …')
        t0 = time.time()
        while not self.ogm.ready:
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
            scan_topic=self.scan_topic,
            cmd_vel_topic=self.cmd_vel_topic,
        )
        if not self.navigator.is_server_ready():
            self.get_logger().error('Nav2 action server not ready!')
            self.coverage_running = False
            return

        rx, ry, ryaw = self._get_robot_pose()
        self.get_logger().info(
            f'NBV start: ({rx:.2f}, {ry:.2f}), sensor_range={self.sensor_range:.2f}m, '
            f'fov={self.sensor_fov_deg:.0f}°, q_min={self.nbv_q_min:.2f}, '
            f'target={self.nbv_target_percent:.1f}%'
        )
        self._reset_coverage_anchor(rx, ry)
        self.nbv_selector.record_observation(rx, ry, ryaw)

        new_samples = self.sampler.generate_samples(rx, ry)
        self.frontier_samples = list(new_samples)
        if new_samples:
            new_ids = self.rcg.expand(new_samples)
            self.rcg.prune(new_ids)
        self.current_node_id = self._nearest_node_id(rx, ry)
        self.initialized = True

        target = float(self.nbv_target_percent)
        no_gain = 0
        failed_frontiers: list[tuple[float, float]] = []
        step = 0

        while self.coverage_running:
            step += 1
            rx, ry, ryaw = self._get_robot_pose()

            # See from here, then check coverage.
            self.nbv_selector.record_observation(rx, ry, ryaw)
            self._publish_observation_grid()
            pct, obs_area, total_free = self.ogm.get_observation_statistics(
                self.nbv_q_min
            )
            self.get_logger().info(
                f'  Step {step}: observed={pct:.1f}% '
                f'({obs_area:.1f}/{total_free:.1f} m²), '
                f'OPEN={self.rcg.num_open}, no_gain={no_gain}'
            )
            if pct >= target:
                self.get_logger().info(
                    f'╔══════════════════════════════════╗\n'
                    f'║   OBSERVED-COVERAGE TARGET MET   ║\n'
                    f'║   {pct:.1f}% ≥ {target:.1f}% — stopping.\n'
                    f'╚══════════════════════════════════╝'
                )
                self.coverage_complete = True
                break

            # Grow candidate viewpoints into newly-revealed / unknown area.
            new_samples = self.sampler.generate_samples(rx, ry)
            if new_samples:
                self.frontier_samples = list(new_samples)
                new_ids = self.rcg.expand(new_samples)
                self.rcg.prune(new_ids)

            next_id, gain = self.nbv_selector.select_view(
                self.current_node_id, rx, ry
            )

            if next_id is None or gain < self.nbv_min_gain:
                # No worthwhile viewpoint — push into unknown via raw frontier,
                # so an unknown place keeps getting explored.
                if self._nbv_push_to_frontier(rx, ry, failed_frontiers):
                    no_gain = 0
                    continue
                no_gain += 1
                if no_gain >= int(self.nbv_no_gain_patience):
                    self.get_logger().info(
                        f'NBV: no information gain for {no_gain} steps and no '
                        f'reachable frontier — stopping at {pct:.1f}%.'
                    )
                    self.coverage_complete = pct >= target
                    break
                time.sleep(0.1)
                continue

            target_node = self.rcg.nodes[next_id]
            self.get_logger().info(
                f'  → view {next_id} ({target_node.x:.2f},{target_node.y:.2f}) '
                f'gain={gain:.1f}'
            )
            self._publish_goal(target_node.x, target_node.y)
            visit_idx = self._log_visit(
                'nbv', target_node.x, target_node.y, node_id=next_id,
            )
            success = self.navigator.go_to(
                target_node.x, target_node.y, prefer_direct=False
            )
            self._mark_visit_result(
                visit_idx, 'arrived' if success else 'nav_failed',
            )

            if not success:
                # Unreachable viewpoint — close it so we stop choosing it.
                if next_id in self.rcg.nodes:
                    self.rcg.set_node_state(next_id, NodeState.CLOSED)
                self.navigator.backup(distance=0.30, speed=0.10, rotate_angle=0.5)
                no_gain += 1
                if no_gain >= int(self.nbv_no_gain_patience):
                    self.get_logger().warn(
                        'NBV: too many unreachable viewpoints — stopping.'
                    )
                    break
                continue

            no_gain = 0
            self.current_node_id = next_id
            self._add_pose(target_node.x, target_node.y)
            self._mark_covered_to(target_node.x, target_node.y)
            rx, ry, ryaw = self._get_robot_pose()
            self.rcg.close_nearby_nodes(rx, ry, self.rc)
            time.sleep(0.05)

        self._publish_observation_grid()
        final_pct, _, _ = self.ogm.get_observation_statistics(self.nbv_q_min)
        self.get_logger().info(
            f'HazMap NBV finished. Observed coverage {final_pct:.1f}% '
            f'in {step} steps.'
        )
        self._save_visit_log()
        self.coverage_running = False

    def _nbv_push_to_frontier(
        self, rx: float, ry: float, failed_frontiers: list
    ) -> bool:
        """Reposition toward the nearest unexplored opening (frontier
        cluster), skipping blacklisted ones, so unknown area keeps getting
        revealed. Returns True if a reposition was attempted successfully."""
        blacklist_radius = max(self.w, 0.50)
        target = self.ogm.find_nearest_frontier_cluster(
            rx, ry,
            min_cluster_cells=3,
            min_distance=max(0.0, self.direct_nav_xy_tolerance),
            exclude=failed_frontiers,
            exclude_radius=blacklist_radius,
        )
        if target is None:
            return False
        tx, ty = self._clamp_goal_to_map(target[0], target[1])
        self.get_logger().info(
            f'  NBV: no graph gain — pushing to frontier ({tx:.2f},{ty:.2f}).'
        )
        visit_idx = self._log_visit('nbv_frontier', tx, ty)
        success = self.navigator.go_to(tx, ty, prefer_direct=False, timeout=60.0)
        self._mark_visit_result(
            visit_idx, 'arrived' if success else 'nav_failed',
        )
        # Blacklist this opening either way: if reached, the next observation
        # reveals it and new frontiers appear elsewhere; if it stays a frontier
        # we must not re-pick the same spot (prevents livelock).
        failed_frontiers.append((tx, ty))
        if success:
            self._add_pose(tx, ty)
            self._mark_covered_to(tx, ty)
            self.current_node_id = self._nearest_node_id(tx, ty)
            return True
        return False

    def _publish_observation_grid(self):
        """Publish the NBV observation-quality field as an OccupancyGrid
        (0..100 = sensing quality) for RViz inspection."""
        grid = self.ogm.observation_quality_grid_int8()
        if grid is None:
            return
        msg = OccupancyGrid()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info.resolution = self.ogm._resolution
        msg.info.width = self.ogm._width
        msg.info.height = self.ogm._height
        msg.info.origin.position.x = self.ogm._origin_x
        msg.info.origin.position.y = self.ogm._origin_y
        msg.info.origin.orientation.w = 1.0
        msg.data = grid.flatten().tolist()
        self.obs_quality_pub.publish(msg)

    # ------------------------------------------------------------------
    # Coverage hole detection (uses C* TSPSolver)
    # ------------------------------------------------------------------
    def _detect_and_cover_holes(self, current_id: int, next_id: int):
        if current_id not in self.rcg.nodes:
            return

        holes = self.tsp_solver.detect_coverage_holes(current_id, next_id)
        if not holes:
            return

        all_hole_nodes = set()
        for h in holes:
            all_hole_nodes.update(h)

        self.get_logger().info(
            f'  Detected {len(holes)} coverage hole(s), '
            f'{len(all_hole_nodes)} total nodes — TSP (Alg 3) …'
        )

        plan = self.tsp_solver.compute_tsp_plan(all_hole_nodes, current_id, next_id)
        if plan is None:
            return

        for wp in plan.waypoints:
            if not self.coverage_running:
                break
            success = self.navigator.go_to(wp.x, wp.y, prefer_direct=False)
            if success:
                self._mark_covered_to(wp.x, wp.y)
                rx, ry = self._robot_x, self._robot_y
                self.rcg.close_nearby_nodes(rx, ry, self.rc)
                self.goal_selector.update_retreat_nodes(rx, ry)
                if wp.node_id is not None and wp.node_id in self.rcg.nodes:
                    self.rcg.set_node_state(wp.node_id, NodeState.CLOSED)
                    self.current_node_id = wp.node_id
                    self._add_pose(wp.x, wp.y)
            else:
                self.get_logger().warn(
                    f'  TSP: failed to reach ({wp.x:.2f},{wp.y:.2f}), skipping.'
                )

    # ------------------------------------------------------------------
    # Graph-path navigation
    # ------------------------------------------------------------------
    def _navigate_graph_path(self, from_id: int, to_id: int):
        """Navigate from from_id to to_id along the RCG.

        Preferred path: hand the whole A* corridor to Nav2 as a single
        NavigateThroughPoses leg — the controller follows the corridor
        smoothly without a plan/arrive round-trip per node. If that's
        unavailable or fails, fall back to a direct goal, then hop-by-hop.
        """
        path = self.rcg.astar(from_id, to_id)
        if path is None:
            path = [to_id]

        # Final destination is the only goal we actively navigate to.
        if to_id not in self.rcg.nodes:
            self.get_logger().warn(f'  Graph-path: target {to_id} missing')
            return
        target = self.rcg.nodes[to_id]

        # Build the corridor waypoints from the A* nodes (drop tiny hops).
        path_poses: list[tuple[float, float]] = []
        last = None
        min_step = max(0.05, 0.25 * self.w)
        for pnid in path:
            n = self.rcg.nodes.get(pnid)
            if n is None:
                continue
            if last is not None and math.hypot(n.x - last[0], n.y - last[1]) < min_step:
                continue
            path_poses.append((n.x, n.y))
            last = (n.x, n.y)

        self.get_logger().info(
            f'  Graph-path {from_id}→{to_id}: {len(path)} hops via '
            f'NavigateThroughPoses ({len(path_poses)} waypoints).'
        )
        if len(path_poses) >= 2 and self.navigator.go_through(
            path_poses, timeout=max(60.0, 20.0 * len(path_poses))
        ):
            for px, py in path_poses:
                self._mark_covered_to(px, py)
                self.rcg.close_nearby_nodes(px, py, self.rc)
            self.rcg.set_node_state(to_id, NodeState.CLOSED)
            self.current_node_id = to_id
            self._add_pose(target.x, target.y)
            rx, ry = self._robot_x, self._robot_y
            self.goal_selector.update_retreat_nodes(rx, ry)
            return

        success = self.navigator.go_to(target.x, target.y, prefer_direct=False)
        if not success:
            self.get_logger().warn(
                f'  Graph-path: direct nav to target {to_id} failed; '
                f'falling back to hop-by-hop.'
            )
            # Fallback: original per-node traversal in case the direct
            # path is blocked but a series of short hops can squeeze through.
            for pnid in path:
                if not self.coverage_running:
                    break
                if pnid not in self.rcg.nodes:
                    continue
                pnode = self.rcg.nodes[pnid]
                ok = self.navigator.go_to(pnode.x, pnode.y, prefer_direct=True)
                if ok:
                    self._mark_covered_to(pnode.x, pnode.y)
                    if pnode.state == NodeState.OPEN:
                        self.rcg.set_node_state(pnid, NodeState.CLOSED)
                    self.current_node_id = pnid
                    self._add_pose(pnode.x, pnode.y)
                    rx, ry = self._robot_x, self._robot_y
                    self.rcg.close_nearby_nodes(rx, ry, self.rc)
                    self.goal_selector.update_retreat_nodes(rx, ry)
            return

        # Direct nav succeeded — close transit nodes whose neighborhood
        # we passed through, and record the arrival.
        self._mark_covered_to(target.x, target.y)
        self.rcg.set_node_state(to_id, NodeState.CLOSED)
        self.current_node_id = to_id
        self._add_pose(target.x, target.y)
        rx, ry = self._robot_x, self._robot_y
        self.rcg.close_nearby_nodes(rx, ry, self.rc)
        self.goal_selector.update_retreat_nodes(rx, ry)

    def _cluster_samples_from_grid(self) -> list:
        """Turn raw-grid frontier clusters into sampler-format tuples
        (x, y, lap_index, lap_position, is_end), so the RCG gets OPEN nodes
        at every opening, even ones the lap-grid sampler misses."""
        if not self.ogm.ready:
            return []
        clusters = self.ogm.find_frontier_clusters(min_cluster_cells=4)
        if not clusters:
            return []
        # Match ProgressiveSampler's lap convention.
        sweep_x = self.sweep_direction == 'x'
        perp_dx, perp_dy = (1.0, 0.0) if sweep_x else (0.0, 1.0)
        sweep_dx, sweep_dy = (0.0, 1.0) if sweep_x else (1.0, 0.0)
        out = []
        for cx, cy, _size in clusters:
            perp_proj = cx * perp_dx + cy * perp_dy
            lap_index = int(round(perp_proj / self.w))
            lap_pos = cx * sweep_dx + cy * sweep_dy
            out.append((cx, cy, lap_index, lap_pos, False))
        return out

    def _inject_region_jump(
        self, rx: float, ry: float, failed_frontiers: list
    ) -> None:
        """Pick the largest frontier cluster that is *not* blacklisted and
        far from the current pose, drive there. Guarantees every distinct
        region gets visited even if local selection is happy looping."""
        clusters = self.ogm.find_frontier_clusters(
            min_cluster_cells=5,
            exclude=failed_frontiers,
            exclude_radius=max(self.w, 0.50),
        )
        if not clusters:
            return
        # Score: prefer large clusters that are far from the rover.
        best = None
        best_score = -1.0
        for cx, cy, size in clusters:
            d = math.hypot(cx - rx, cy - ry)
            score = float(size) * d
            if score > best_score:
                best_score = score
                best = (cx, cy, size)
        if best is None:
            return
        tx, ty = self._clamp_goal_to_map(best[0], best[1])
        self.get_logger().info(
            f'  Region injector → cluster size={best[2]} '
            f'({tx:.2f},{ty:.2f}).'
        )
        visit_idx = self._log_visit('region_jump', tx, ty)
        ok = self.navigator.go_to(
            tx, ty, prefer_direct=False, timeout=120.0,
        )
        self._mark_visit_result(
            visit_idx, 'arrived' if ok else 'nav_failed',
        )
        if ok:
            self._add_pose(tx, ty)
            self._mark_covered_to(tx, ty)
            rx2, ry2 = self._robot_x, self._robot_y
            self.rcg.close_nearby_nodes(rx2, ry2, self.rc)
            nearest = self._nearest_node_id(tx, ty)
            if nearest is not None:
                self.current_node_id = nearest
        else:
            failed_frontiers.append((tx, ty))

    def _drive_route_ntp(self, waypoints, chunk_size: int = 25) -> int:
        """Drive a precomputed waypoint route in NavigateThroughPoses chunks.

        Returns the number of waypoints completed before the first failure (or
        the full count). The caller resumes per-waypoint navigation from there,
        so the robust single-goal loop stays as the fallback."""
        completed = 0
        n = len(waypoints)
        while completed < n and self.coverage_running:
            chunk = waypoints[completed:completed + chunk_size]
            poses = [self._clamp_goal_to_map(wx, wy) for wx, wy in chunk]
            if not self.navigator.go_through(
                poses, timeout=max(60.0, 15.0 * len(poses))
            ):
                break
            for px, py in poses:
                self._add_pose(px, py)
                self._mark_covered_to(px, py)
            completed += len(chunk)
        return completed

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _nearest_node_id(self, x: float, y: float):
        best_id = None
        best_d = float('inf')
        for nid, n in self.rcg.nodes.items():
            d = math.hypot(n.x - x, n.y - y)
            if d < best_d:
                best_d = d
                best_id = nid
        return best_id

    def _nearest_open_node_euclidean(self, x: float, y: float):
        best = None
        best_d = float('inf')
        for nid in self.rcg._open_ids:
            n = self.rcg.nodes.get(nid)
            if n is None:
                continue
            d = math.hypot(n.x - x, n.y - y)
            if d < best_d:
                best_d = d
                best = n
        return best

    def _get_robot_pose(self):
        return (self._robot_x, self._robot_y, self._robot_yaw)

    def _reset_coverage_anchor(self, x: float, y: float):
        """Reset segment-marking anchor for continuous coverage attribution."""
        self._coverage_anchor = (x, y)
        self.sampler.mark_covered(x, y, self.rc)

    def _mark_covered_to(self, x: float, y: float):
        """
        Mark coverage along the traversed segment from last anchor to (x, y).
        This avoids under-counting area when waypoints are sparse.
        """
        if self._coverage_anchor is None:
            self._reset_coverage_anchor(x, y)
            return

        x0, y0 = self._coverage_anchor
        dist = math.hypot(x - x0, y - y0)
        step = max(0.05, float(self.rc) * 0.6)
        samples = max(1, int(math.ceil(dist / step)))
        for i in range(1, samples + 1):
            t = i / float(samples)
            sx = x0 + (x - x0) * t
            sy = y0 + (y - y0) * t
            self.sampler.mark_covered(sx, sy, self.rc)
        self._coverage_anchor = (x, y)

    def _clamp_goal_to_map(self, x: float, y: float):
        """Clamp goals to valid map interior to avoid planner edge failures."""
        if not self.ogm.ready:
            return x, y
        m = max(self.w, self.rc, 0.30)
        xmin = self.ogm._origin_x + m
        ymin = self.ogm._origin_y + m
        xmax = self.ogm._origin_x + self.ogm._width * self.ogm._resolution - m
        ymax = self.ogm._origin_y + self.ogm._height * self.ogm._resolution - m
        return (min(max(x, xmin), xmax), min(max(y, ymin), ymax))

    def _coverage_stats(self):
        """Return current coverage stats as (percent, area_m2, total_m2)."""
        return self.ogm.get_coverage_statistics()

    def _refine_known_map_coverage(self):
        """
        In known-map mode, keep visiting nearest uncovered FREE cells until
        coverage_target_percent is reached or progress stalls.
        """
        if not self.known_map_mode or self.navigator is None:
            p, _, _ = self._coverage_stats()
            return p

        target = float(self.coverage_target_percent)
        percent, _, _ = self._coverage_stats()
        if percent >= target:
            return percent

        attempts              = 0
        consecutive_nav_fails = 0
        total_nav_fails       = 0          # never resets — hard ceiling
        max_goals             = int(self.coverage_refine_max_goals)
        max_consec_failures   = int(self.coverage_refine_max_nav_failures)
        max_total_failures    = int(self.coverage_refine_total_failures_limit)
        timeout_sec           = float(self.coverage_refine_timeout_sec)
        search_range          = float(self.coverage_refine_search_range)
        no_gain_patience      = max(1, int(self.coverage_refine_no_gain_patience))
        min_gain              = max(0.0, float(self.coverage_refine_min_gain_percent))
        stagnation_count      = 0
        best_percent          = percent
        t_start               = time.time()

        self.get_logger().info(
            f'Coverage refinement: {percent:.1f}% → target {target:.1f}%  '
            f'(max_goals={max_goals}, timeout={timeout_sec:.0f}s)'
        )

        while self.coverage_running and attempts < max_goals and percent < target:

            # ── wall-clock timeout ────────────────────────────────────────────
            elapsed = time.time() - t_start
            if elapsed >= timeout_sec:
                self.get_logger().warn(
                    f'Coverage refinement timed out after {elapsed:.0f}s '
                    f'at {percent:.1f}%.'
                )
                break

            rx, ry, _ = self._get_robot_pose()
            goal = self.ogm.find_nearest_uncovered_free(
                rx, ry,
                max_range=search_range,
                min_distance=max(0.0, self.direct_nav_xy_tolerance),
            )
            if goal is None:
                self.get_logger().warn(
                    'Coverage refinement: no uncovered free cell found — done.'
                )
                break

            tx, ty = self._clamp_goal_to_map(goal[0], goal[1])
            attempts += 1
            self._publish_goal(tx, ty)
            ok = self.navigator.go_to(tx, ty, prefer_direct=True, timeout=30.0)
            if not ok:
                consecutive_nav_fails += 1
                total_nav_fails       += 1
                self.get_logger().warn(
                    f'Coverage refinement: nav fail ({tx:.2f},{ty:.2f})  '
                    f'consec={consecutive_nav_fails}/{max_consec_failures}  '
                    f'total={total_nav_fails}/{max_total_failures}'
                )
                if consecutive_nav_fails >= max_consec_failures:
                    self.get_logger().warn(
                        'Coverage refinement: consecutive failure limit reached.'
                    )
                    break
                if total_nav_fails >= max_total_failures:
                    self.get_logger().warn(
                        'Coverage refinement: total failure limit reached.'
                    )
                    break
                self.navigator.backup(distance=0.20, speed=0.08, rotate_angle=0.35)
                continue

            consecutive_nav_fails = 0
            self._add_pose(tx, ty)
            self._mark_covered_to(tx, ty)
            percent, _, _ = self._coverage_stats()
            gain = percent - best_percent
            if gain < min_gain:
                stagnation_count += 1
            else:
                best_percent  = percent
                stagnation_count = 0

            elapsed = time.time() - t_start
            if attempts % 3 == 0 or percent >= target:
                self.get_logger().info(
                    f'  coverage={percent:.1f}%  target={target:.1f}%  '
                    f'attempts={attempts}  elapsed={elapsed:.0f}s'
                )
            if stagnation_count >= no_gain_patience:
                self.get_logger().warn(
                    f'Coverage refinement: stagnated for {stagnation_count} goals '
                    f'at {percent:.1f}% — stopping.'
                )
                break

        elapsed = time.time() - t_start
        self.get_logger().info(
            f'Coverage refinement finished: {percent:.1f}%  '
            f'attempts={attempts}  elapsed={elapsed:.0f}s'
        )
        return percent

    def _direct_safety_check(self, rx: float, ry: float, tx: float, ty: float) -> bool:
        if not self.ogm.ready:
            return False
        return self.ogm.is_collision_free(rx, ry, tx, ty)

    def _should_prefer_direct_transition(self, from_id: int, to_id: int) -> bool:
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
        if not self.ogm.is_collision_free(src.x, src.y, dst.x, dst.y):
            return False
        src_clear = self.ogm.nearest_obstacle_distance(src.x, src.y)
        dst_clear = self.ogm.nearest_obstacle_distance(dst.x, dst.y)
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

    def _log_visit(
        self,
        source: str,
        target_x: float,
        target_y: float,
        node_id=None,
        result: str = 'pending',
    ) -> int:
        """Record a goal choice. Call once per goal attempt.

        source : "cstar" | "deadend_escape" | "outer_repos" |
                 "safety_net" | "refine" | "stc" | "boustro"
        result : "pending" at goal-issue; update with mark_visit_result(idx, ...).
        Returns the row index in self._visit_log so the caller can update it.
        """
        self._visit_step += 1
        pct, area, total = self._coverage_stats()
        row = {
            'time':            f'{time.time():.3f}',
            'step':            self._visit_step,
            'source':          source,
            'node_id':         '' if node_id is None else int(node_id),
            'target_x':        f'{target_x:.3f}',
            'target_y':        f'{target_y:.3f}',
            'robot_x':         f'{self._robot_x:.3f}',
            'robot_y':         f'{self._robot_y:.3f}',
            'result':          result,
            'coverage_pct':    f'{pct:.2f}',
            'area_covered_m2': f'{area:.2f}',
            'total_free_m2':   f'{total:.2f}',
            'rcg_nodes':       len(self.rcg.nodes),
            'rcg_open':        self.rcg.num_open,
        }
        self._visit_log.append(row)
        return len(self._visit_log) - 1

    def _mark_visit_result(self, idx: int, result: str) -> None:
        """Update the result of a previously-logged visit row."""
        if 0 <= idx < len(self._visit_log):
            self._visit_log[idx]['result'] = result
            pct, area, total = self._coverage_stats()
            self._visit_log[idx]['coverage_pct']    = f'{pct:.2f}'
            self._visit_log[idx]['area_covered_m2'] = f'{area:.2f}'
            self._visit_log[idx]['total_free_m2']   = f'{total:.2f}'

    def _save_visit_log(self) -> None:
        """Save the visit log as CSV alongside any other results."""
        if not self._visit_log:
            self.get_logger().info('No visits to log.')
            return
        import csv as _csv
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        out_dir = os.path.expanduser(f'~/ros2_ws/results/{timestamp}')
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as e:
            self.get_logger().warn(f'Cannot create results dir {out_dir}: {e}')
            out_dir = os.path.expanduser('~')
        path = os.path.join(out_dir, 'visit_log.csv')
        fieldnames = list(self._visit_log[0].keys())
        try:
            with open(path, 'w', newline='') as f:
                w = _csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(self._visit_log)
            self.get_logger().info(
                f'Visit log: {len(self._visit_log)} rows → {path}'
            )
        except Exception as e:
            self.get_logger().warn(f'Failed to write visit log: {e}')

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

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    def _publish_viz(self):
        if not self.initialized:
            return
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

        for node in list(self.rcg.nodes.values()):
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
            elif node.id in self.goal_selector.retreat_nodes:
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
            t.text = f'L{node.lap_index}'
            ma.markers.append(t)

        self.nodes_pub.publish(ma)

    def _pub_frontier_points(self, stamp):
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
            m.pose.position.x = fs[0]
            m.pose.position.y = fs[1]
            m.pose.position.z = 0.05
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.04
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
        for a, b in list(self.rcg.edges):
            na = self.rcg.nodes.get(a)
            nb = self.rcg.nodes.get(b)
            if na is None or nb is None:
                continue
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = stamp
            m.ns = 'rcg_edges'
            m.id = eid
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.points = [
                Point(x=na.x, y=na.y, z=0.05),
                Point(x=nb.x, y=nb.y, z=0.05),
            ]
            m.scale.x = 0.015
            if na.lap_index == nb.lap_index:
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
        is_x = (self.sweep_direction == 'x')

        laps: dict = {}
        for n in list(self.rcg.nodes.values()):
            lid = n.lap_index
            if lid not in laps:
                laps[lid] = {'pos': n.x if is_x else n.y,
                             'mn': n.y if is_x else n.x,
                             'mx': n.y if is_x else n.x}
            else:
                if is_x:
                    laps[lid]['mn'] = min(laps[lid]['mn'], n.y)
                    laps[lid]['mx'] = max(laps[lid]['mx'], n.y)
                else:
                    laps[lid]['mn'] = min(laps[lid]['mn'], n.x)
                    laps[lid]['mx'] = max(laps[lid]['mx'], n.x)

        mid = 1
        for li, info in laps.items():
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = stamp
            m.ns = 'laps'
            m.id = mid
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
            mid += 1
        self.laps_pub.publish(ma)

    def _record_trajectory(self):
        if not self.coverage_running:
            return
        rx, ry, ryaw = self._robot_x, self._robot_y, self._robot_yaw
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = rx
        pose.pose.position.y = ry
        pose.pose.orientation.z = math.sin(ryaw / 2.0)
        pose.pose.orientation.w = math.cos(ryaw / 2.0)
        self.trajectory_poses.append(pose)

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
        node._save_visit_log()
        node.coverage_running = False
        node.destroy_node()
        # Launch may already have shut down the shared context on SIGINT.
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
