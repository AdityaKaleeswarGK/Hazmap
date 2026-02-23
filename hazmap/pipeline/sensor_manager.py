import csv
import hashlib
import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
from scipy import interpolate
from scipy.ndimage import gaussian_filter

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
    return _DEFAULT_SENSOR_PLOT_CMAP


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


def _create_smooth_heatmap_grid(
    readings: List[Tuple[float, float, float]],
    x_min: float, y_min: float, x_max: float, y_max: float,
    resolution: float, smooth_sigma: float = 3.0,
) -> Optional[np.ndarray]:
    if len(readings) < 4:
        return None
    
    points = np.array([(r[0], r[1]) for r in readings])
    values = np.array([r[2] for r in readings])
    
    cols = max(1, int((x_max - x_min) / resolution))
    rows = max(1, int((y_max - y_min) / resolution))
    
    xi = np.linspace(x_min, x_max, cols)
    yi = np.linspace(y_min, y_max, rows)
    grid_x, grid_y = np.meshgrid(xi, yi)
    
    try:
        rbf = interpolate.Rbf(
            points[:, 0], points[:, 1], values,
            function='thin_plate', smooth=0.1,
        )
        grid = rbf(grid_x, grid_y)
    except Exception:
        grid = interpolate.griddata(
            points, values, (grid_x, grid_y), method='cubic'
        )
        nan_mask = np.isnan(grid)
        if nan_mask.any():
            nearest = interpolate.griddata(
                points, values, (grid_x, grid_y), method='nearest'
            )
            grid[nan_mask] = nearest[nan_mask]
    
    if smooth_sigma > 0:
        grid = gaussian_filter(grid, sigma=smooth_sigma, mode='nearest')
    
    return grid


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

    def get_interpolated_grid(self, smooth_sigma: float = 2.0) -> Optional[np.ndarray]:
        if len(self.readings) < 3:
            return None

        mask = self.count_grid > 0
        grid = np.full((self.rows, self.cols), np.nan)
        grid[mask] = self.sum_grid[mask] / self.count_grid[mask]

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

        if smooth_sigma > 0:
            grid = gaussian_filter(grid, sigma=smooth_sigma, mode='nearest')

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
    def _zscore_grid_to_01(grid: np.ndarray, std_clip: float = 3.0) -> Optional[np.ndarray]:
        """Converts raw readings to a Z-score anomaly map (0.0 to 1.0)."""
        mean_val = float(np.nanmean(grid))
        std_val = float(np.nanstd(grid))
        if std_val <= 1e-12:
            return np.zeros_like(grid) # Uniform room, no anomalies
        
        # Calculate Z-Scores
        z_scores = (grid - mean_val) / std_val
        
        # We only care about positive anomalies (spikes above the room average).
        # Clip anything below average to 0.0, and cap extreme spikes at std_clip.
        z_scores_clipped = np.clip(z_scores, 0.0, std_clip)
        
        # Normalize the 0->std_clip range to 0.0->1.0 for the colormap.
        return z_scores_clipped / std_clip

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

    def _create_map_aligned_grid(self, readings: List[Tuple[float, float, float]],
                                 smooth_sigma: float = 2.0) -> Optional[np.ndarray]:
        """Interpolates sensor readings directly onto the map manager's occupancy grid."""
        mm = self.node.map_manager
        if mm.occupancy_grid is None or len(readings) < 3:
            return None

        xi = np.linspace(mm.origin_x, mm.origin_x + mm.grid_width * mm.resolution, mm.grid_width)
        yi = np.linspace(mm.origin_y, mm.origin_y + mm.grid_height * mm.resolution, mm.grid_height)
        grid_x, grid_y = np.meshgrid(xi, yi)

        points = np.array([(r[0], r[1]) for r in readings])
        values = np.array([r[2] for r in readings])

        grid = np.full((mm.grid_height, mm.grid_width), np.nan)

        interp = interpolate.griddata(points, values, (grid_x, grid_y), method='linear')
        grid[np.isnan(grid)] = interp[np.isnan(grid)]

        still_nan = np.isnan(grid)
        if still_nan.any():
            nearest = interpolate.griddata(points, values, (grid_x, grid_y), method='nearest')
            grid[still_nan] = nearest[still_nan]

        if smooth_sigma > 0:
            grid = gaussian_filter(grid, sigma=smooth_sigma, mode='nearest')

        return grid

    def save_results(self, logger, detection_manager=None):
        """Generate perfectly map-aligned heatmap PNGs."""
        if self._results_saved:
            return
        self._results_saved = True

        has_sensor_data = any(len(sim.readings) > 0 for sim in self.simulators.values())
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
            import copy
        except ImportError:
            logger.error('matplotlib not installed — cannot save heatmaps. Install with: pip install matplotlib')
            return

        plt.style.use('default')

        file_dir = os.path.dirname(os.path.abspath(__file__))
        ws_root = _find_ros_workspace_root(file_dir)
        if ws_root is not None:
            base_results_dir = os.path.join(ws_root, 'results')
        else:
            project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            base_results_dir = os.path.join(project_dir, 'results')

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = os.path.join(base_results_dir, timestamp)
        os.makedirs(results_dir, exist_ok=True)

        logger.info(f'Saving HazMap results to {results_dir}')

        csv_path = os.path.join(results_dir, 'gas_concentrations.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['sensor', 'x', 'y', 'concentration'])
            for name, sim in self.simulators.items():
                for x, y, val in sim.readings:
                    writer.writerow([name, f'{x:.4f}', f'{y:.4f}', f'{val:.6f}'])
        logger.info(f'  Saved {csv_path}')

        mm = self.node.map_manager
        if mm.occupancy_grid is None:
            logger.warn('Occupancy grid is missing, cannot generate visual maps.')
            return

        map_extent = [
            mm.origin_x, mm.origin_x + mm.grid_width * mm.resolution,
            mm.origin_y, mm.origin_y + mm.grid_height * mm.resolution,
        ]

        map_img = np.full(mm.occupancy_grid.shape, 1.0)  # Unknown is white to remove grey borders
        map_img[mm.occupancy_grid == 0] = 1.0            # Free is white
        map_img[mm.occupancy_grid >= 50] = 0.0           # Obstacle is black

        free_mask = (mm.occupancy_grid == 0)  # Interpolate gas fully across known free space

        def add_scale_and_stats(ax_obj, raw_grid, free_space_mask, extent, unit_label=''):
            raw_masked = np.where(free_space_mask, raw_grid, np.nan)
            valid = raw_masked[~np.isnan(raw_masked)]
            if len(valid) > 0:
                v_min, v_max, v_mean = np.nanmin(valid), np.nanmax(valid), np.nanmean(valid)
                v_std = np.nanstd(valid)
                ul = f" ({unit_label})" if unit_label else ""
                stats_text = f"Stats{ul}:\nMin: {v_min:.2f}\nMax: {v_max:.2f}\nMean: {v_mean:.2f}\nStd: {v_std:.2f}"
                props = dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='#cbd5e1')
                ax_obj.text(0.02, 0.98, stats_text, transform=ax_obj.transAxes, fontsize=10,
                        verticalalignment='top', bbox=props, zorder=10)
            
            map_w = extent[1] - extent[0]
            scale_length = 1.0 if map_w < 10 else 5.0
            if map_w >= 20: scale_length = 10.0
            
            sx = extent[0] + map_w * 0.05
            sy = extent[2] + (extent[3] - extent[2]) * 0.05
            tick_h = map_w * 0.01
            
            ax_obj.plot([sx, sx + scale_length], [sy, sy], color='black', linewidth=3, zorder=10)
            ax_obj.plot([sx, sx], [sy - tick_h, sy + tick_h], color='black', linewidth=1.5, zorder=10)
            ax_obj.plot([sx + scale_length, sx + scale_length], [sy - tick_h, sy + tick_h], color='black', linewidth=1.5, zorder=10)
            ax_obj.text(sx + scale_length/2, sy + tick_h * 1.5, f'{scale_length} m', 
                    color='black', fontsize=10, ha='center', va='bottom', fontweight='bold', zorder=10,
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.1))

        grids_for_consolidated = {}
        for name, sim in self.simulators.items():
            if len(sim.readings) < 3:
                continue

            grid = self._create_map_aligned_grid(sim.readings)
            if grid is None:
                continue

            normalized = self._zscore_grid_to_01(grid)
            if normalized is None:
                continue

            grids_for_consolidated[name] = normalized

            fig = plt.figure(figsize=(10, 10), facecolor='white')
            ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
            ax.set_facecolor('white')

            sensor_cmap = copy.copy(plt.get_cmap(_get_sensor_plot_cmap(name)))
            sensor_cmap.set_bad(color='white', alpha=0.0)

            ax.imshow(map_img, origin='lower', extent=map_extent, cmap='gray', vmin=0.0, vmax=1.0)
            masked_grid = np.where(free_mask, normalized, np.nan)

            im = ax.imshow(
                masked_grid, origin='lower', extent=map_extent,
                cmap=sensor_cmap, vmin=0.0, vmax=1.0,
                aspect='equal', interpolation='bilinear',
            )

            # --- TOPOGRAPHIC CONTOURS ---
            if np.nanmax(masked_grid) > 0.1:
                ax.contour(
                    masked_grid, levels=6, origin='lower', extent=map_extent,
                    colors='black', alpha=0.35, linewidths=0.8
                )
            
            add_scale_and_stats(ax, grid, free_mask, map_extent, sim.config.unit)

            ax.set_title(f'HazMap — {name.upper()} Concentration Map\n(Z-Score Anomaly)', pad=15)
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            _apply_light_plot_style(ax)

            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            unit_label = f' ({sim.config.unit})' if sim.config.unit else ''
            cbar.set_label(f'Concentration{unit_label}', rotation=270, labelpad=15)

            path = os.path.join(results_dir, f'{name}_heatmap.png')
            fig.savefig(path, dpi=200, pad_inches=0, facecolor='white')
            plt.close(fig)
            logger.info(f'  Saved {path}')

        if has_detection_data:
            self._save_detection_pins(detection_manager, results_dir, logger, map_extent, map_img)

        if grids_for_consolidated or has_detection_data:
            # Professional academic continuous heat palette
            colors_list = [
                '#ffffff', '#e0f2fe', '#7dd3fc', '#0ea5e9', '#4f46e5',
                '#7e22ce', '#d946ef', '#f43f5e', '#f97316', '#eab308'
            ]
            impact_cmap = LinearSegmentedColormap.from_list('impact', colors_list, N=256)
            self._save_consolidated(
                grids_for_consolidated, results_dir, impact_cmap, logger,
                map_extent, map_img, free_mask, detection_manager=detection_manager
            )
            self._save_connectivity_graph_figure(
                results_dir, logger, map_extent, map_img, detection_manager=detection_manager
            )

    def _save_detection_pins(self, det_mgr, results_dir, logger, map_extent, map_img):
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        confirmed = det_mgr.get_confirmed_detections()
        if not confirmed:
            return

        fig, ax = plt.subplots(1, 1, figsize=(12, 10), facecolor='white')
        ax.imshow(map_img, origin='lower', extent=map_extent, cmap='gray', vmin=0.0, vmax=1.0)

        class_groups: dict = {}
        for det in confirmed:
            class_groups.setdefault(det.cls_name, []).append(det)

        legend_handles = []
        for cls_name, dets in class_groups.items():
            color = det_mgr._get_class_color(cls_name)
            xs = [d.map_x for d in dets]
            ys = [d.map_y for d in dets]
            sizes = [30 + min(d.detection_count, 20) * 5 for d in dets]
            
            ax.scatter(
                xs, ys, s=sizes, c=[color],
                edgecolors='black', linewidths=0.8,
                zorder=5, alpha=0.9,
            )
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

    def _save_connectivity_graph_figure(self, results_dir, logger, map_extent, map_img, detection_manager=None):
        import matplotlib.pyplot as plt

        rcg = self.node.rcg
        if not rcg.nodes:
            return

        fig, ax = plt.subplots(1, 1, figsize=(12, 10), facecolor='white')
        ax.imshow(map_img, origin='lower', extent=map_extent, cmap='gray', vmin=0.0, vmax=1.0)

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

    def _save_consolidated(
        self, grids, results_dir, cmap, logger,
        map_extent, map_img, free_mask, detection_manager=None, smooth_sigma: float = 2.5
    ):
        import matplotlib.pyplot as plt
        import copy

        total_weight = 0.0
        combined = None

        for name, grid in grids.items():
            cfg = self.simulators[name].config
            weight = 1.0 / cfg.priority
            total_weight += weight

            if combined is None:
                combined = weight * grid
            else:
                combined += weight * grid

        if detection_manager is not None and detection_manager.enabled:
            confirmed = detection_manager.get_confirmed_detections()
            if confirmed:
                mm = self.node.map_manager
                det_grid = detection_manager.get_detection_impact_grid(
                    map_extent[0], map_extent[2],
                    mm.grid_width, mm.grid_height,
                    mm.resolution
                )
                if det_grid is not None:
                    avg_pri = np.mean([d.priority for d in confirmed])
                    det_weight = 1.0 / max(avg_pri, 1.0)
                    total_weight += det_weight

                    if combined is None:
                        combined = det_weight * det_grid
                    else:
                        combined += det_weight * det_grid

        if combined is None or total_weight == 0:
            return

        combined /= total_weight

        fig = plt.figure(figsize=(10, 10), facecolor='white')
        ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
        ax.set_facecolor('white')
        
        cmap = copy.copy(cmap)
        cmap.set_bad(color='white', alpha=0.0)
        
        if smooth_sigma > 0:
            combined = gaussian_filter(combined, sigma=smooth_sigma, mode='nearest')
        
        ax.imshow(map_img, origin='lower', extent=map_extent, cmap='gray', vmin=0.0, vmax=1.0)
        
        masked_combined = np.where(free_mask, combined, np.nan)
        
        max_comb = np.nanmax(masked_combined)
        dyn_vmax = max(0.1, float(max_comb))
        
        im = ax.imshow(
            masked_combined, origin='lower', extent=map_extent,
            cmap=cmap, vmin=0.0, vmax=dyn_vmax,
            aspect='equal', interpolation='bilinear',
        )
        
        if max_comb > 0.1:
            ax.contour(
                masked_combined, levels=10, origin='lower', extent=map_extent,
                colors='black', alpha=0.4, linewidths=0.6
            )
            
        mm = self.node.map_manager
        
        # Draw Priority-Based Peak Markers
        peaks = []
        for name, grid in grids.items():
            cfg = self.simulators[name].config
            valid_grid = np.where(free_mask, grid, np.nan)
            if np.nanmax(valid_grid) > 0.1:
                max_idx = np.nanargmax(valid_grid)
                flat_y, flat_x = np.unravel_index(max_idx, valid_grid.shape)
                cx = map_extent[0] + (flat_x / mm.grid_width) * (map_extent[1] - map_extent[0])
                cy = map_extent[2] + (flat_y / mm.grid_height) * (map_extent[3] - map_extent[2])
                peaks.append({'label': name.upper(), 'x': cx, 'y': cy, 'pri': cfg.priority})
                
        if detection_manager is not None and detection_manager.enabled:
            dets = detection_manager.get_confirmed_detections()
            if dets:
                det_grid = detection_manager.get_detection_impact_grid(
                    map_extent[0], map_extent[2], mm.grid_width, mm.grid_height, mm.resolution
                )
                if det_grid is not None:
                    valid_det = np.where(free_mask, det_grid, np.nan)
                    if np.nanmax(valid_det) > 0.0:
                        max_idx = np.nanargmax(valid_det)
                        flat_y, flat_x = np.unravel_index(max_idx, valid_det.shape)
                        cx = map_extent[0] + (flat_x / mm.grid_width) * (map_extent[1] - map_extent[0])
                        cy = map_extent[2] + (flat_y / mm.grid_height) * (map_extent[3] - map_extent[2])
                        avg_pri = np.mean([d.priority for d in dets])
                        peaks.append({'label': 'DETECTION', 'x': cx, 'y': cy, 'pri': avg_pri})
                
        # Sort so lower priority (higher number) is drawn first, high priority on top
        peaks.sort(key=lambda p: p['pri'], reverse=True)
        
        # Plot peaks with shades based on priority (1 = red, 2 = orange, 3 = yellow, 4+ = cyan)
        priority_colors = {1: '#e31a1c', 2: '#f97316', 3: '#eab308'}
        
        for p in peaks:
            # Determine color and size by priority
            pri_int = max(1, int(round(p['pri'])))
            p_color = priority_colors.get(pri_int, '#06b6d4')  # default cyan for low priority
            p_size = max(6, 16 - pri_int * 2)  # PRI 1 = 14, PRI 2 = 12, etc.
            
            # Draw marker
            ax.plot(p['x'], p['y'], marker='o', color=p_color, markersize=p_size, 
                    markeredgecolor='black', markeredgewidth=1.2, zorder=20)
            
            # Draw label (Cleaner styling for academic paper)
            bbox_props = dict(boxstyle="round,pad=0.25", fc="white", ec='black', alpha=0.85, lw=0.8)
            ax.text(p['x'], p['y'] + (map_extent[3]-map_extent[2])*0.025, f"{p['label']}", 
                    color='black', fontsize=8, fontweight='bold', ha='center', va='bottom', 
                    bbox=bbox_props, zorder=25)

        # Add scale bar
        map_w = map_extent[1] - map_extent[0]
        scale_length = 1.0 if map_w < 10 else 5.0
        if map_w >= 20: scale_length = 10.0
        sx = map_extent[0] + map_w * 0.05
        sy = map_extent[2] + (map_extent[3] - map_extent[2]) * 0.05
        tick_h = map_w * 0.01
        ax.plot([sx, sx + scale_length], [sy, sy], color='black', linewidth=3, zorder=10)
        ax.plot([sx, sx], [sy - tick_h, sy + tick_h], color='black', linewidth=1.5, zorder=10)
        ax.plot([sx + scale_length, sx + scale_length], [sy - tick_h, sy + tick_h], color='black', linewidth=1.5, zorder=10)
        ax.text(sx + scale_length/2, sy + tick_h * 1.5, f'{scale_length} m', 
                color='black', fontsize=10, ha='center', va='bottom', fontweight='bold', zorder=10,
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.1))
        
        ax.set_title('HazMap — Consolidated Hazard Map\n(Actionable Danger Zones)', pad=15)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        _apply_light_plot_style(ax)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Relative Hazard Impact Severity', rotation=270, labelpad=15)

        path = os.path.join(results_dir, 'consolidated_impact.png')
        fig.savefig(path, dpi=200, pad_inches=0, facecolor='white')
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

