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
from .navigator import Navigator


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
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('auto_start_coverage', False)
        # ── Coverage-saturation pruning (shift from node-exhaustion) ─────
        # Periodically CLOSE OPEN nodes that sit in already-covered, fully
        # known space (low coverage gain AND not bordering unknown). This
        # stops the rover from crossing the covered map just to "tick off"
        # redundant nodes — the main source of late-game trajectory overlap.
        # Frontier nodes (adjacent to unknown) are always kept so genuine
        # unexplored regions still get a goal.
        self.declare_parameter('prune_enable', True)
        self.declare_parameter('prune_interval', 4)         # steps between sweeps
        self.declare_parameter('prune_gain_min_cells', 15)  # < this = redundant
        self.declare_parameter('prune_keep_frontier', True)
        # ── Phase 1: local commitment + stuck-node safety ────────────────
        # commit_threshold: pick a direct lap neighbor unless a far candidate
        # has > 1/commit_threshold times more uncovered area. 0.30 = need
        # >3.3× more gain to deviate (paper-faithful local sweep behavior).
        # Set to 1.0 to disable (pure utility argmax = Phase-0 behavior).
        self.declare_parameter('commit_threshold', 0.30)
        # Phase-6 path-cost penalty: when the straight line from current to
        # candidate is collision-blocked, multiply Euclidean cost by this
        # factor in the utility score. Demotes Euclidean-near but path-far
        # candidates that would force a long Nav2 detour (the step-18 "0.85m
        # goal that's actually 10m around the wall" pattern). 1.0 = disabled.
        self.declare_parameter('path_blocked_penalty', 3.0)
        # Phase-7 per-step Nav2 timeout. Caps how long a single Nav2 goal
        # may run before we give up and let the failure path (backup + retry,
        # then close as unreachable after MAX_NODE_FAILS) take over. The
        # 206 s step-40 stall last sprint was Nav2 saturating its 20Hz
        # controller and burning the default 120 s timeout twice (2×120≈240).
        # Default 50 s caps worst case at ~100 s before the node is dropped.
        self.declare_parameter('nav_step_timeout_s', 50.0)
        # If the same OPEN node is selected this many times in a row without
        # the rover physically getting within rc of it, force-close it so
        # the run can't livelock on an unreachable target (e.g. the corner
        # node that sits 0.02 m outside reach inside an inflated wall).
        self.declare_parameter('stuck_node_patience', 3)
        # Late-game termination guard: if a node is *re*-selected this many
        # times across the whole inner loop (not just consecutively) without
        # ever closing, force-close it. Catches the "ring of unreachable
        # frontier nodes cycling forever" pattern at the end of a run.
        self.declare_parameter('stuck_node_total_patience', 3)
        # ── Phase 2: en-route node closing ───────────────────────────────
        # After each traverse, CLOSE OPEN nodes the rover physically drove
        # over (within enroute_close_radius of the path segment) and that are
        # now covered — so it never deliberately drives back to a node it
        # already covered in transit. Frontier nodes are kept.
        self.declare_parameter('enroute_close_enable', True)
        self.declare_parameter('enroute_close_radius', 0.55)
        # ── Phase 4: adaptive frontier density (surveillance framing) ────
        # In open empty regions (>= density_open_distance from any obstacle),
        # keep only density_keep_floor fraction of lap samples. Cluttered /
        # feature-rich regions stay densely sampled. Set keep_floor=1.0 to
        # disable. The selector then has FEWER scattered open-area targets
        # to chase, killing the residual long jumps to sparse open nodes.
        self.declare_parameter('density_keep_floor', 0.40)
        self.declare_parameter('density_open_distance', 1.5)
        # ── Unreachable-node pruning ─────────────────────────────────────
        # Close OPEN nodes lodged inside the costmap inflation (too close to
        # an obstacle for the rover to ever reach). These never close on
        # arrival and cause cross-map ping-pong as the selector keeps
        # re-targeting them — the main end-game overlap source.
        self.declare_parameter('unreachable_prune_enable', True)
        self.declare_parameter('unreachable_obstacle_margin', 0.28)
        # ── Coverage-efficiency early stop ───────────────────────────────
        # Stop when area-gained-per-metre over a rolling window falls below
        # eff_stop_threshold AND coverage already exceeds eff_stop_min_coverage.
        # Robust to the stagnation detector being fooled by transit coverage.
        self.declare_parameter('eff_stop_enable', True)
        self.declare_parameter('eff_window_size', 6)
        self.declare_parameter('eff_stop_threshold', 0.12)   # m² per metre
        self.declare_parameter('eff_stop_min_coverage', 80.0)  # percent

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
        self.map_topic = self.get_parameter('map_topic').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.scan_topic = self.get_parameter('scan_topic').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.auto_start_coverage = self.get_parameter('auto_start_coverage').value
        self.prune_enable = self.get_parameter('prune_enable').value
        self.prune_interval = max(1, int(self.get_parameter('prune_interval').value))
        self.prune_gain_min_cells = int(
            self.get_parameter('prune_gain_min_cells').value
        )
        self.prune_keep_frontier = self.get_parameter('prune_keep_frontier').value
        self.commit_threshold = float(
            self.get_parameter('commit_threshold').value
        )
        self.path_blocked_penalty = float(
            self.get_parameter('path_blocked_penalty').value
        )
        self.nav_step_timeout_s = float(
            self.get_parameter('nav_step_timeout_s').value
        )
        self.stuck_node_patience = int(
            self.get_parameter('stuck_node_patience').value
        )
        self.stuck_node_total_patience = int(
            self.get_parameter('stuck_node_total_patience').value
        )
        self.enroute_close_enable = self.get_parameter(
            'enroute_close_enable'
        ).value
        self.enroute_close_radius = float(
            self.get_parameter('enroute_close_radius').value
        )
        self.density_keep_floor = float(
            self.get_parameter('density_keep_floor').value
        )
        self.density_open_distance = float(
            self.get_parameter('density_open_distance').value
        )
        self.unreachable_prune_enable = self.get_parameter(
            'unreachable_prune_enable'
        ).value
        self.unreachable_obstacle_margin = float(
            self.get_parameter('unreachable_obstacle_margin').value
        )
        self.eff_stop_enable = self.get_parameter('eff_stop_enable').value
        self.eff_window_size = max(
            2, int(self.get_parameter('eff_window_size').value)
        )
        self.eff_stop_threshold = float(
            self.get_parameter('eff_stop_threshold').value
        )
        self.eff_stop_min_coverage = float(
            self.get_parameter('eff_stop_min_coverage').value
        )

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
            known_map_mode=False,
            density_keep_floor=self.density_keep_floor,
            density_open_distance=self.density_open_distance,
        )
        self.rcg = RCG(self.w, self.ogm)
        self.goal_selector = GoalSelector(
            self.rcg, ogm=self.ogm, rc=self.rc,
            commit_threshold=self.commit_threshold,
            path_blocked_penalty=self.path_blocked_penalty,
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
            f'rd={self.rd}m, sweep={self.sweep_direction}'
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

        self.get_logger().info('Initial 360° rotation for SLAM discovery skipped.')

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

        # Early termination: stop when coverage reaches this %. Set >100 to disable.
        early_stop_coverage = 95.0

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
        # Coverage-efficiency early stop: rolling window of (area_gained,
        # distance_travelled) per step. When coverage-per-metre drops low
        # AND we're already substantially covered, the rover is just dashing
        # across covered map to pick off scattered wall fragments — stop.
        # This is robust to the stagnation detector being fooled by the
        # coverage a long transit paints en route.
        eff_window: list = []  # list of (area_gained, distance)
        prev_eff_rx, prev_eff_ry = robot_x, robot_y
        # Track repeated selection of the same OPEN node when the rover
        # physically can't get within rc of it (e.g. corner nodes lodged
        # inside the costmap inflation). Force-close after N reselects.
        last_selected_id = None
        same_id_streak = 0
        node_visit_counts: dict = {}

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

                # Coverage-saturation pruning: drop OPEN nodes sitting in
                # already-covered, fully-known space so the selector never
                # crosses the map to "tick off" redundant nodes. Done before
                # selection (we've already arrived; current node is CLOSED).
                if self.prune_enable and (step % self.prune_interval == 1):
                    n_pruned = self._prune_redundant_open_nodes()
                    if n_pruned and self.debug:
                        self.get_logger().info(
                            f'  Pruned {n_pruned} redundant OPEN nodes '
                            f'(covered & known); OPEN now {self.rcg.num_open}'
                        )

                if self.rcg.num_open == 0:
                    self.get_logger().info(
                        'No productive OPEN nodes remain '
                        '(all covered or pruned).'
                    )
                    break

                next_id = self.goal_selector.select_goal_node(self.current_node_id)

                # Stuck-node safety: two paths.
                # (a) Consecutive reselects of the same node → force-close after
                #     stuck_node_patience (handles the node-111 corner case).
                # (b) Total reselects of the *same* node across the whole inner
                #     loop → force-close after stuck_node_total_patience. This
                #     catches the late-game pattern where the selector cycles
                #     through a small ring of physically-unreachable frontier
                #     nodes (last sprint: 6 OPEN nodes at run end, none ever
                #     hit the consecutive-3 threshold, so the run hung).
                if next_id is not None:
                    node_visit_counts[next_id] = (
                        node_visit_counts.get(next_id, 0) + 1
                    )
                    if (node_visit_counts[next_id]
                            >= self.stuck_node_total_patience):
                        self.get_logger().warn(
                            f'  Node {next_id} selected '
                            f'{node_visit_counts[next_id]}× total without '
                            f'closing — force-closing as unreachable.'
                        )
                        self.rcg.set_node_state(next_id, NodeState.CLOSED)
                        last_selected_id = None
                        same_id_streak = 0
                        node_visit_counts.pop(next_id, None)
                        continue
                if (next_id is not None and next_id == last_selected_id
                        and self.current_node_id != next_id):
                    same_id_streak += 1
                    if same_id_streak >= self.stuck_node_patience:
                        self.get_logger().warn(
                            f'  Stuck on node {next_id} after '
                            f'{same_id_streak} reselects — force-closing.'
                        )
                        self.rcg.set_node_state(next_id, NodeState.CLOSED)
                        last_selected_id = None
                        same_id_streak = 0
                        continue  # re-enter loop top, will reselect
                else:
                    same_id_streak = 0
                last_selected_id = next_id

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

                target = self.rcg.nodes[next_id]
                self.get_logger().info(
                    f'  Step {step}: {self.current_node_id}→{next_id} '
                    f'L{target.lap_index} ({target.x:.2f},{target.y:.2f})'
                )
                self._publish_goal(target.x, target.y)
                visit_idx = self._log_visit(
                    'cstar', target.x, target.y, node_id=next_id,
                )

                success = self.navigator.go_to(
                    target.x, target.y, prefer_direct=False,
                    timeout=self.nav_step_timeout_s,
                )
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

                # Mark coverage swath. Capture the segment we just traversed
                # BEFORE _mark_covered_to advances the anchor, so we can also
                # close OPEN nodes the rover physically drove past en route —
                # not just those near the arrival pose. This stops the rover
                # from later "re-visiting" a node it already covered in transit
                # (the node-92-twice pattern at the end of the last sprint).
                seg_x0, seg_y0 = (
                    self._coverage_anchor
                    if self._coverage_anchor is not None
                    else (arrived.x, arrived.y)
                )
                self._mark_covered_to(arrived.x, arrived.y)

                rx, ry = self._robot_x, self._robot_y
                self.rcg.close_nearby_nodes(rx, ry, self.rc)
                if self.enroute_close_enable:
                    n_enroute = self._close_nodes_along_path(
                        seg_x0, seg_y0, arrived.x, arrived.y
                    )
                    if n_enroute and self.debug:
                        self.get_logger().info(
                            f'  En-route: closed {n_enroute} OPEN nodes the '
                            f'rover passed over in transit.'
                        )
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
                # Region jump only when we're actually stuck (no recent
                # coverage progress), not on a fixed timer — a periodic
                # cross-map dash was itself a big overlap contributor. When
                # the local selector is productive, leave it alone.
                region_inject_counter += 1
                if (stagnation_count >= 2
                        and region_inject_counter >= REGION_INJECT_INTERVAL):
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

                # Coverage-efficiency tracking (area gained per metre moved).
                eff_dist = math.hypot(rx - prev_eff_rx, ry - prev_eff_ry)
                prev_eff_rx, prev_eff_ry = rx, ry
                eff_window.append((max(0.0, area_growth), eff_dist))
                if len(eff_window) > self.eff_window_size:
                    eff_window.pop(0)

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

                # Coverage-efficiency early stop. Only consider it once we're
                # past min coverage and the window is full, so a single far
                # jump can't trigger it — sustained low yield does.
                if (self.eff_stop_enable
                        and pct >= self.eff_stop_min_coverage
                        and len(eff_window) >= self.eff_window_size):
                    win_area = sum(a for a, _ in eff_window)
                    win_dist = sum(d for _, d in eff_window)
                    efficiency = win_area / win_dist if win_dist > 1e-3 else 0.0
                    if efficiency < self.eff_stop_threshold:
                        self.get_logger().info(
                            f'╔══════════════════════════════════╗\n'
                            f'║   DIMINISHING RETURNS — STOPPING ║\n'
                            f'║   {win_area:.2f} m² gained over '
                            f'{win_dist:.1f} m (eff {efficiency:.3f} < '
                            f'{self.eff_stop_threshold:.3f} m²/m)\n'
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

            self.get_logger().info(
                'Inner loop done — 360° rotation for SLAM discovery skipped.'
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
                                timeout=self.nav_step_timeout_s,
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
            # Clamp cluster centroid to inside-map bounds. Frontier
            # clustering can return cells right at the boundary; the sampler
            # then places a candidate that sits *outside* the navigable map
            # (e.g. node 99 at y=4.53 when map top was ~4.0 in the last
            # sprint), which Nav2 can never reach → nav_failed loop.
            cx, cy = self._clamp_goal_to_map(cx, cy)
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

    def _prune_redundant_open_nodes(self) -> int:
        """Close OPEN nodes that are no longer worth visiting: their local
        area is already covered (coverage gain below threshold) AND they do
        not border unknown space. Frontier nodes (adjacent to unknown) are
        always kept so genuine unexplored regions still get a goal.

        This is the core of the coverage-saturation strategy: instead of
        driving to *every* node, the rover only chases nodes that still add
        new coverage or that lead into the unknown. Returns the count closed.
        """
        if not self.prune_enable or not self.ogm.ready:
            return 0
        radius = max(self.rc, 0.5 * self.w)
        pruned = 0
        for nid in list(self.rcg._open_ids):
            node = self.rcg.nodes.get(nid)
            if node is None:
                continue
            # Never prune the node we're currently sitting on.
            if nid == self.current_node_id:
                continue
            # Unreachable-node prune: a node inside the costmap inflation
            # (too close to an obstacle for the rover to reach) will never
            # close — it just causes cross-map ping-pong as the selector
            # re-targets it. Close it regardless of frontier status. This is
            # the main fix for the wall-lodged-node end-game overlap.
            if self.unreachable_prune_enable:
                d_obs = self.ogm.nearest_obstacle_distance(node.x, node.y)
                if d_obs < self.unreachable_obstacle_margin:
                    self.rcg.set_node_state(nid, NodeState.CLOSED)
                    pruned += 1
                    continue
            if self.prune_keep_frontier and self.ogm.is_adjacent_to_unknown(
                node.x, node.y, self.w
            ):
                continue
            gain = self.ogm.predict_coverage_gain(node.x, node.y, radius)
            if gain < self.prune_gain_min_cells:
                self.rcg.set_node_state(nid, NodeState.CLOSED)
                pruned += 1
        return pruned

    @staticmethod
    def _point_segment_distance(px, py, x0, y0, x1, y1) -> float:
        """Shortest distance from point (px,py) to segment (x0,y0)-(x1,y1)."""
        dx, dy = x1 - x0, y1 - y0
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq <= 1e-9:
            return math.hypot(px - x0, py - y0)
        t = ((px - x0) * dx + (py - y0) * dy) / seg_len_sq
        t = max(0.0, min(1.0, t))
        cx, cy = x0 + t * dx, y0 + t * dy
        return math.hypot(px - cx, py - cy)

    def _close_nodes_along_path(self, x0, y0, x1, y1) -> int:
        """Close OPEN nodes whose position lies within the coverage swath of
        the segment the rover just traversed — i.e. nodes it physically drove
        over. These were already covered in transit, so deliberately driving
        back to them later is pure overlap. Frontier nodes (adjacent to
        unknown) are kept so genuine exploration targets survive.

        Returns the count of nodes closed.
        """
        if not self.ogm.ready:
            return 0
        radius = self.enroute_close_radius
        closed = 0
        for nid in list(self.rcg._open_ids):
            node = self.rcg.nodes.get(nid)
            if node is None or nid == self.current_node_id:
                continue
            d = self._point_segment_distance(node.x, node.y, x0, y0, x1, y1)
            if d > radius:
                continue
            # Keep frontier nodes: bordering unknown means there may still be
            # area to *discover* there even if we drove past the free side.
            if self.prune_keep_frontier and self.ogm.is_adjacent_to_unknown(
                node.x, node.y, self.w
            ):
                continue
            # Only close if it's actually covered now (gain ~0) — guards
            # against closing a node that sits near the path but still fronts
            # a genuinely uncovered pocket the straight-segment swath missed.
            gain = self.ogm.predict_coverage_gain(
                node.x, node.y, max(self.rc, 0.5 * self.w)
            )
            if gain < self.prune_gain_min_cells:
                self.rcg.set_node_state(nid, NodeState.CLOSED)
                closed += 1
        return closed

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
