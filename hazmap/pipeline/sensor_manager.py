import hashlib
import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
from scipy import interpolate

from visualization_msgs.msg import Marker, MarkerArray


@dataclass
class SensorConfig:
    """Definition of one environmental sensor."""
    name: str
    unit: str
    priority: int
    topic: str = ''

    def __post_init__(self):
        if not self.topic:
            self.topic = f'/hazmap/{self.name}_markers'


_SENSOR_SIM_RANGES: dict = {
    'co':          (0.0,   200.0),
    'co2':         (350.0, 5000.0),
    'methane':     (0.0,   500.0),
    'o2':          (15.0,  21.0),
    'temperature': (20.0,  80.0),
    'ch4':         (0.0,   500.0),
}
_DEFAULT_SIM_RANGE = (0.0, 100.0)

_SENSOR_PLOT_CMAPS: dict = {
    'co':          'Reds',
    'co2':         'Greens',
    'methane':     'PuRd',
    'o2':          'Blues_r',
    'temperature': 'inferno',
    'ch4':         'PuRd',
}
_DEFAULT_SENSOR_PLOT_CMAP = 'viridis'


def _find_ros_workspace_root(start_path: str) -> Optional[str]:
    """Walk upward to find a ROS2 workspace root (e.g. */ros2_ws)."""
    current = os.path.abspath(start_path)
    while True:
        if (
            os.path.basename(current) == 'ros2_ws'
            and os.path.isdir(os.path.join(current, 'src'))
        ):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            return None
        current = parent


def _get_sensor_plot_cmap(sensor_name: str) -> str:
    """Return a per-sensor colormap for saved heatmaps."""
    return _SENSOR_PLOT_CMAPS.get(sensor_name.lower(), _DEFAULT_SENSOR_PLOT_CMAP)


def _apply_light_plot_style(ax):
    """Force a readable light style independent of user matplotlib theme."""
    ax.set_facecolor('white')
    ax.grid(True, color='#d1d5db', alpha=0.30, linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_color('#6b7280')
    ax.tick_params(colors='#111827')
    ax.xaxis.label.set_color('#111827')
    ax.yaxis.label.set_color('#111827')
    ax.title.set_color('#111827')


def _grid_to_mesh(extent, grid):
    rows, cols = grid.shape
    xi = np.linspace(extent[0], extent[1], cols)
    yi = np.linspace(extent[2], extent[3], rows)
    return np.meshgrid(xi, yi)


class SensorSimulator:
    """Simulates readings for a single sensor and accumulates a grid."""

    def __init__(self, config: SensorConfig, grid_resolution: float,
                 splash_radius: float,
                 x_min: float, y_min: float, x_max: float, y_max: float):
        self.config = config
        self.resolution = grid_resolution
        self.splash_radius = splash_radius

        sim_range = _SENSOR_SIM_RANGES.get(config.name, _DEFAULT_SIM_RANGE)
        self.sim_min = sim_range[0]
        self.sim_max = sim_range[1]
        self._splash_cells = max(1, int(splash_radius / grid_resolution))

        self.x_min = x_min - 0.5
        self.y_min = y_min - 0.5
        self.x_max = x_max + 0.5
        self.y_max = y_max + 0.5

        self.cols = max(1, int((self.x_max - self.x_min) / self.resolution))
        self.rows = max(1, int((self.y_max - self.y_min) / self.resolution))

        self.sum_grid = np.zeros((self.rows, self.cols), dtype=np.float64)
        self.count_grid = np.zeros((self.rows, self.cols), dtype=np.int32)

        self.readings: List[Tuple[float, float, float]] = []

        self._marker_id = 0

        self.hotspots = self._generate_hotspots()

    def _generate_hotspots(self) -> list:
        seed_hash = int(hashlib.md5(self.config.name.encode()).hexdigest(), 16)
        rng = np.random.RandomState(seed_hash % (2**31))

        n_hotspots = rng.randint(3, 7)
        hotspots = []
        for _ in range(n_hotspots):
            cx = rng.uniform(self.x_min + 0.5, self.x_max - 0.5)
            cy = rng.uniform(self.y_min + 0.5, self.y_max - 0.5)
            sigma = rng.uniform(0.5, 2.0)
            intensity = rng.uniform(0.3, 1.0)
            hotspots.append((cx, cy, sigma, intensity))
        return hotspots

    def read_value(self, x: float, y: float) -> float:
        val_range = self.sim_max - self.sim_min
        base = self.sim_min + 0.2 * val_range
        spatial = 0.05 * val_range * (
            math.sin(x * 1.5) * math.cos(y * 1.5)
        )
        hotspot_val = 0.0
        for cx, cy, sigma, intensity in self.hotspots:
            dist_sq = (x - cx) ** 2 + (y - cy) ** 2
            hotspot_val += intensity * val_range * math.exp(
                -dist_sq / (2.0 * sigma ** 2)
            )
        noise = np.random.normal(0, 0.02 * val_range)
        value = base + spatial + hotspot_val + noise
        return float(np.clip(value, self.sim_min, self.sim_max))

    def record_reading(self, x: float, y: float, value: float):
        center_col = int((x - self.x_min) / self.resolution)
        center_row = int((y - self.y_min) / self.resolution)
        r = self._splash_cells

        for dr in range(-r, r + 1):
            for dc in range(-r, r + 1):
                row = center_row + dr
                col = center_col + dc
                if not (0 <= row < self.rows and 0 <= col < self.cols):
                    continue
                dist = math.sqrt(dr * dr + dc * dc) * self.resolution
                if dist > self.splash_radius:
                    continue
                weight = math.exp(
                    -dist * dist / (2.0 * (self.splash_radius * 0.5) ** 2)
                )
                self.sum_grid[row, col] += value * weight
                self.count_grid[row, col] += 1

        self.readings.append((x, y, value))

    def get_interpolated_grid(self) -> Optional[np.ndarray]:
        if len(self.readings) < 3:
            return None

        mask = self.count_grid > 0
        grid = np.full((self.rows, self.cols), np.nan)
        grid[mask] = self.sum_grid[mask] / self.count_grid[mask]

        if np.isnan(grid).any():
            xi = np.linspace(self.x_min, self.x_max, self.cols)
            yi = np.linspace(self.y_min, self.y_max, self.rows)
            grid_x, grid_y = np.meshgrid(xi, yi)

            points = np.array([(r[0], r[1]) for r in self.readings])
            values = np.array([r[2] for r in self.readings])

            interp = interpolate.griddata(
                points, values, (grid_x, grid_y), method='linear'
            )
            nan_mask = np.isnan(grid)
            grid[nan_mask] = interp[nan_mask]

            still_nan = np.isnan(grid)
            if still_nan.any():
                nearest = interpolate.griddata(
                    points, values, (grid_x, grid_y), method='nearest'
                )
                grid[still_nan] = nearest[still_nan]

        return grid

    @staticmethod
    def value_to_rgb(normalized: float) -> Tuple[float, float, float]:
        n = max(0.0, min(1.0, normalized))
        if n < 0.5:
            t = n / 0.5
            return (t, 1.0, 0.0)
        else:
            t = (n - 0.5) / 0.5
            return (1.0, 1.0 - t, 0.0)

    def create_marker(self, x: float, y: float, value: float,
                      stamp) -> Marker:
        vals = [r[2] for r in self.readings]
        lo = min(vals) if vals else value
        hi = max(vals) if vals else value
        val_range = hi - lo
        normalized = (value - lo) / val_range if val_range > 0 else 0.0

        r, g, b = self.value_to_rgb(normalized)
        diameter = self.splash_radius * 2.0

        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = stamp
        m.ns = f'{self.config.name}_sensor'
        m.id = self._marker_id
        self._marker_id += 1
        m.type = Marker.CYLINDER
        m.action = Marker.ADD
        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = 0.01
        m.pose.orientation.w = 1.0
        m.scale.x = diameter
        m.scale.y = diameter
        m.scale.z = 0.02
        m.color.r = float(r)
        m.color.g = float(g)
        m.color.b = float(b)
        m.color.a = 0.35
        m.lifetime.sec = 0
        return m


class SensorManager:
    """Manages all sensors: sampling, publishing markers, saving results."""

    def __init__(self, node, configs: List[SensorConfig],
                 grid_resolution: float, splash_radius: float):
        self.node = node
        self.configs = configs
        self._results_saved = False

        mm = node.map_manager
        if mm.map_received:
            x_min = mm.origin_x
            y_min = mm.origin_y
            x_max = mm.origin_x + mm.grid_width * mm.resolution
            y_max = mm.origin_y + mm.grid_height * mm.resolution
        else:
            x_min, y_min, x_max, y_max = -5.0, -5.0, 5.0, 5.0

        self.simulators: dict = {}
        self.publishers: dict = {}

        for cfg in configs:
            self.simulators[cfg.name] = SensorSimulator(
                cfg, grid_resolution, splash_radius,
                x_min, y_min, x_max, y_max,
            )
            self.publishers[cfg.name] = node.create_publisher(
                MarkerArray, cfg.topic, 10
            )

    @staticmethod
    def _normalize_grid_01(grid: np.ndarray) -> Optional[np.ndarray]:
        gmin = float(np.nanmin(grid))
        gmax = float(np.nanmax(grid))
        val_range = gmax - gmin
        if val_range <= 1e-12:
            return None
        return (grid - gmin) / val_range

    def sample_and_publish(self):
        if not self.node.coverage_running:
            return

        rx = self.node.map_manager.robot_x
        ry = self.node.map_manager.robot_y
        stamp = self.node.get_clock().now().to_msg()

        for name, sim in self.simulators.items():
            value = sim.read_value(rx, ry)
            sim.record_reading(rx, ry, value)

            marker = sim.create_marker(rx, ry, value, stamp)
            ma = MarkerArray()
            ma.markers = [marker]
            self.publishers[name].publish(ma)

    # ════════════════════════════════════════════════════════════════
    #  save_results — now accepts an optional DetectionManager
    # ════════════════════════════════════════════════════════════════

    def save_results(self, logger, detection_manager=None):
        """Generate heatmap PNGs.  Guarded against double-save."""
        if self._results_saved:
            return
        self._results_saved = True

        has_sensor_data = any(
            len(sim.readings) > 0 for sim in self.simulators.values()
        )
        has_detection_data = (
            detection_manager is not None
            and detection_manager.enabled
            and len(detection_manager.get_confirmed_detections()) > 0
        )

        if not has_sensor_data and not has_detection_data:
            logger.info('No sensor / detection data — skipping save.')
            return

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.colors import LinearSegmentedColormap
        except ImportError:
            logger.error(
                'matplotlib not installed — cannot save heatmaps. '
                'Install with: pip install matplotlib'
            )
            return

        plt.style.use('default')

        file_dir = os.path.dirname(os.path.abspath(__file__))
        ws_root = _find_ros_workspace_root(file_dir)
        if ws_root is not None:
            base_results_dir = os.path.join(ws_root, 'results')
        else:
            project_dir = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            base_results_dir = os.path.join(project_dir, 'results')

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = os.path.join(base_results_dir, timestamp)
        os.makedirs(results_dir, exist_ok=True)

        logger.info(f'Saving HazMap results to {results_dir}')

        colors_list = ['#fee5d9', '#fcae91', '#fb6a4a', '#cb181d', '#67000d']
        impact_cmap = LinearSegmentedColormap.from_list(
            'impact', colors_list, N=256
        )

        # ── Per-sensor heatmaps ─────────────────────────────────────
        grids_for_consolidated = {}
        for name, sim in self.simulators.items():
            grid = sim.get_interpolated_grid()
            if grid is None:
                logger.warn(
                    f'Sensor "{name}": not enough data for heatmap.'
                )
                continue

            normalized = self._normalize_grid_01(grid)
            if normalized is None:
                logger.warn(
                    f'Sensor "{name}": degenerate grid range; skipping.'
                )
                continue

            grids_for_consolidated[name] = normalized

            fig, ax = plt.subplots(1, 1, figsize=(11, 9), facecolor='white')
            sensor_cmap = plt.get_cmap(_get_sensor_plot_cmap(name))
            extent = [sim.x_min, sim.x_max, sim.y_min, sim.y_max]
            X, Y = _grid_to_mesh(extent, normalized)
            levels = np.linspace(0.0, 1.0, 20)
            cf = ax.contourf(
                X, Y, normalized, levels=levels, cmap=sensor_cmap,
                vmin=0.0, vmax=1.0, antialiased=True, alpha=0.95,
            )
            cbar = plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(f'{sim.config.name.upper()} (normalized 0-1)')
            ax.set_title(
                f'{sim.config.name.upper()} Concentration Heatmap (0-1)'
            )
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            self._draw_obstacle_outline(ax)

            vals = normalized[np.isfinite(normalized)]
            if vals.size:
                stats = (
                    f"Min: {float(np.min(vals)):.2f}\n"
                    f"Max: {float(np.max(vals)):.2f}\n"
                    f"Mean: {float(np.mean(vals)):.2f}"
                )
                ax.text(
                    0.015, 0.97, stats, transform=ax.transAxes, va='top',
                    fontsize=9,
                    bbox=dict(
                        boxstyle='round', facecolor='white', alpha=0.75
                    ),
                )
            _apply_light_plot_style(ax)

            path = os.path.join(results_dir, f'{name}_heatmap.png')
            fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            logger.info(f'  Saved {path}')

        # ── Detection pin graph ─────────────────────────────────────
        if has_detection_data:
            self._save_detection_pins(
                detection_manager, results_dir, logger,
            )

        # ── Consolidated + connectivity ─────────────────────────────
        if grids_for_consolidated or has_detection_data:
            self._save_consolidated(
                grids_for_consolidated, results_dir, impact_cmap, logger,
                detection_manager=detection_manager,
            )
            self._save_connectivity_graph_figure(
                results_dir, logger,
                detection_manager=detection_manager,
            )

    # ─── Detection pin graph ────────────────────────────────────────

    def _save_detection_pins(self, det_mgr, results_dir, logger):
        """Save a map-view figure with coloured pins for each detection."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.lines import Line2D
        except ImportError:
            return

        confirmed = det_mgr.get_confirmed_detections()
        if not confirmed:
            return

        fig, ax = plt.subplots(1, 1, figsize=(12, 10), facecolor='white')
        self._draw_obstacle_outline(ax)

        # Group by class for legend
        class_groups: dict = {}
        for det in confirmed:
            class_groups.setdefault(det.cls_name, []).append(det)

        legend_handles = []
        for cls_name, dets in class_groups.items():
            color = det_mgr._get_class_color(cls_name)
            xs = [d.map_x for d in dets]
            ys = [d.map_y for d in dets]
            sizes = [
                30 + min(d.detection_count, 20) * 5 for d in dets
            ]
            ax.scatter(
                xs, ys, s=sizes, c=[color],
                edgecolors='black', linewidths=0.8,
                zorder=5, alpha=0.9,
            )
            # Splash circles
            for d in dets:
                circle = plt.Circle(
                    (d.map_x, d.map_y), det_mgr.splash_radius,
                    color=color, alpha=0.15, linewidth=0.5,
                    edgecolor=color, linestyle='--',
                )
                ax.add_patch(circle)

            legend_handles.append(
                Line2D(
                    [0], [0], marker='o', color='w',
                    markerfacecolor=color, markeredgecolor='black',
                    markersize=10,
                    label=f'{cls_name} ({len(dets)})',
                )
            )

        ax.set_title('HazMap — Visual Detection Pins')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        if legend_handles:
            ax.legend(handles=legend_handles, loc='upper right', framealpha=0.9)
        ax.set_aspect('equal', adjustable='datalim')
        _apply_light_plot_style(ax)

        path = os.path.join(results_dir, 'detection_pins.png')
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        logger.info(f'  Saved {path}')

    # ─── Obstacle outline helper ────────────────────────────────────

    def _draw_obstacle_outline(self, ax):
        mm = self.node.map_manager
        if mm.occupancy_grid is None:
            return
        obstacle = (mm.occupancy_grid >= 50).astype(float)
        extent = [
            mm.origin_x,
            mm.origin_x + mm.grid_width * mm.resolution,
            mm.origin_y,
            mm.origin_y + mm.grid_height * mm.resolution,
        ]
        X, Y = _grid_to_mesh(extent, obstacle)
        ax.contour(
            X, Y, obstacle, levels=[0.5],
            colors=['#9ca3af'], linewidths=0.8, alpha=0.75,
        )

    # ─── Connectivity graph ─────────────────────────────────────────

    def _save_connectivity_graph_figure(
        self, results_dir, logger, detection_manager=None,
    ):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            return

        rcg = self.node.rcg
        if not rcg.nodes:
            return

        fig, ax = plt.subplots(1, 1, figsize=(12, 10), facecolor='white')
        self._draw_obstacle_outline(ax)

        drawn = set()
        for node in rcg.nodes.values():
            for nbid in node.neighbors.keys():
                key = (min(node.id, nbid), max(node.id, nbid))
                if key in drawn or nbid not in rcg.nodes:
                    continue
                drawn.add(key)
                nb = rcg.nodes[nbid]
                ax.plot(
                    [node.x, nb.x], [node.y, nb.y],
                    color='#cbd5e1', linewidth=0.8, alpha=0.85, zorder=1,
                )

        node_x = [n.x for n in rcg.nodes.values()]
        node_y = [n.y for n in rcg.nodes.values()]
        ax.scatter(
            node_x, node_y, s=26, c='#ef4444',
            edgecolors='white', linewidths=0.4, zorder=3,
        )

        visited = self.node.visited_poses
        if len(visited) >= 2:
            px = [p.pose.position.x for p in visited]
            py = [p.pose.position.y for p in visited]
            ax.plot(
                px, py, color='#d946ef', linewidth=2.0, alpha=0.9,
                zorder=4, label='Executed path',
            )
            ax.scatter(
                px[0], py[0], s=55, c='#16a34a', zorder=5, label='Start',
            )
            ax.scatter(
                px[-1], py[-1], s=55, c='#dc2626', zorder=5, label='End',
            )

        # Overlay detection pins on connectivity graph
        if detection_manager is not None and detection_manager.enabled:
            dets = detection_manager.get_confirmed_detections()
            if dets:
                for det in dets:
                    color = detection_manager._get_class_color(det.cls_name)
                    ax.scatter(
                        det.map_x, det.map_y, s=60,
                        c=[color], marker='D',
                        edgecolors='black', linewidths=0.6,
                        zorder=6, alpha=0.9,
                    )

        ax.set_title('HazMap — Coverage Node Connectivity Graph')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.legend(loc='upper right', framealpha=0.9)
        _apply_light_plot_style(ax)

        path = os.path.join(results_dir, 'coverage_connectivity_graph.png')
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        logger.info(f'  Saved {path}')

    # ─── Consolidated impact map ────────────────────────────────────

    def _save_consolidated(
        self, grids, results_dir, cmap, logger,
        detection_manager=None,
    ):
        """Weighted combination of sensor + detection data with danger contours."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        total_weight = 0.0
        combined = None

        # ── Sensor grids ──
        for name, grid in grids.items():
            cfg = self.simulators[name].config
            weight = 1.0 / cfg.priority
            total_weight += weight

            if combined is None:
                combined = weight * grid
            else:
                min_rows = min(combined.shape[0], grid.shape[0])
                min_cols = min(combined.shape[1], grid.shape[1])
                combined = combined[:min_rows, :min_cols]
                combined += weight * grid[:min_rows, :min_cols]

        # ── Detection impact grid ──
        if detection_manager is not None and detection_manager.enabled:
            confirmed = detection_manager.get_confirmed_detections()
            if confirmed:
                ref_sim = next(iter(self.simulators.values()), None)
                if ref_sim is not None:
                    det_grid = detection_manager.get_detection_impact_grid(
                        ref_sim.x_min, ref_sim.y_min,
                        ref_sim.x_max, ref_sim.y_max,
                    )
                    if det_grid is not None:
                        avg_pri = np.mean(
                            [d.priority for d in confirmed]
                        )
                        det_weight = 1.0 / max(avg_pri, 1.0)
                        total_weight += det_weight

                        if combined is None:
                            combined = det_weight * det_grid
                        else:
                            min_rows = min(
                                combined.shape[0], det_grid.shape[0]
                            )
                            min_cols = min(
                                combined.shape[1], det_grid.shape[1]
                            )
                            combined = combined[:min_rows, :min_cols]
                            combined += (
                                det_weight
                                * det_grid[:min_rows, :min_cols]
                            )

        if combined is None or total_weight == 0:
            return

        combined /= total_weight

        ref_sim = next(iter(self.simulators.values()))
        extent = [ref_sim.x_min, ref_sim.x_max, ref_sim.y_min, ref_sim.y_max]

        fig, ax = plt.subplots(1, 1, figsize=(14, 12), facecolor='white')
        X, Y = _grid_to_mesh(extent, combined)
        levels = np.linspace(0.0, 1.0, 24)
        cf = ax.contourf(
            X, Y, combined, levels=levels, cmap=cmap,
            vmin=0.0, vmax=1.0, antialiased=True, alpha=0.96,
        )

        contour_levels = [0.5, 0.7, 0.9]
        contour_colors = ['#fb6a4a', '#cb181d', '#67000d']
        cs = ax.contour(
            X, Y, combined, levels=contour_levels,
            colors=contour_colors, linewidths=1.2,
        )
        ax.clabel(cs, fmt={0.5: '50%', 0.7: '70%', 0.9: '90%'}, fontsize=9)

        # Overlay detection pins on the consolidated map
        if detection_manager is not None and detection_manager.enabled:
            dets = detection_manager.get_confirmed_detections()
            for det in dets:
                color = detection_manager._get_class_color(det.cls_name)
                ax.scatter(
                    det.map_x, det.map_y, s=50,
                    c=[color], marker='v',
                    edgecolors='black', linewidths=0.6,
                    zorder=10, alpha=0.95,
                )
                # Splash circle
                circle = plt.Circle(
                    (det.map_x, det.map_y),
                    detection_manager.splash_radius,
                    color=color, alpha=0.12, linewidth=0.5,
                    edgecolor=color, linestyle='--',
                )
                ax.add_patch(circle)

        cbar = plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Weighted Hazard Impact Level (0-1)')
        ax.set_title('HazMap — Consolidated Hazard Impact Map (normalized 0-1)')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        self._draw_obstacle_outline(ax)
        _apply_light_plot_style(ax)

        # Legend info
        info_parts = []
        for cfg in self.configs:
            if cfg.name in grids:
                info_parts.append(f'{cfg.name}(w={1.0/cfg.priority:.2f})')
        if detection_manager is not None and detection_manager.enabled:
            n_det = len(detection_manager.get_confirmed_detections())
            if n_det > 0:
                info_parts.append(f'detections({n_det} objects)')
        sensor_info = ', '.join(info_parts)
        ax.text(
            0.02, 0.02, f'Sources: {sensor_info}',
            transform=ax.transAxes, fontsize=7,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
        )

        path = os.path.join(results_dir, 'consolidated_impact.png')
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        logger.info(f'  Saved {path}')


def load_sensor_configs_from_params(node) -> List[SensorConfig]:
    node.declare_parameter('sensor_names', ['co', 'co2', 'methane', 'o2'])
    node.declare_parameter('sensor_units', ['ppm', 'ppm', 'ppm', '%'])
    node.declare_parameter('sensor_priorities', [1, 2, 3, 4])

    names = node.get_parameter('sensor_names').value
    units = node.get_parameter('sensor_units').value
    priorities = node.get_parameter('sensor_priorities').value

    configs = []
    for i, name in enumerate(names):
        cfg = SensorConfig(
            name=name,
            unit=units[i] if i < len(units) else '',
            priority=priorities[i] if i < len(priorities) else (i + 1),
        )
        configs.append(cfg)

    return configs

