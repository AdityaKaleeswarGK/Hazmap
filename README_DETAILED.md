# HazMap — Technical Architecture & Pipeline Analysis

This document provides a detailed technical breakdown of the core algorithms and pipelines used in the HazMap system for autonomous hazard mapping.

---

## 1. Gas Localization & Environmental Sampling

The gas localization pipeline is responsible for simulating, sampling, and processing environmental sensor data (CO, CO₂, Methane, O₂) to identify potential leaks or hazardous concentrations.

### Sampling Pipeline
1.  **Simulation**: Controlled by `SensorSimulator`, which uses a combination of base concentration, spatial sin/cos variations, and Gaussian-distributed hotspots to simulate realistic gas dispersal.
    ```python
    value = base + spatial + hotspot_val + noise
    ```
2.  **Recording (Gaussian Splash)**: When a sensor reading is taken, it isn't just recorded as a point. It is recorded as a **Gaussian Splash** on a local grid. This accounts for the sensor's range and the physical dispersal of gases.
    ```python
    weight = math.exp(-dist * dist / (2.0 * (splash_radius * 0.5) ** 2))
    sim.sum_grid[row, col] += value * weight
    ```
3.  **Z-Score Anomaly Detection**: To differentiate actual leaks from background noise, the system calculates a Z-score for every grid cell. This normalizes the data relative to the average concentration in the room.
    -   **Z-Score Calculation**: `z = (x - mean) / std`
    -   Anomalies are clipped to a specific standard deviation (e.g., 3.0) and mapped to a [0, 1] range for visualization.

---

## 2. Reachability Connectivity Graph (RCG)

The RCG is the core data structure driving the coverage path planning algorithm. It organizes the explored space into a navigable graph.

### Graph Generation & Expansion
1.  **Frontier Sampling**: `ProgressiveSampler` identifies the boundary between explored and unknown space (the "frontier"). It generates sample points along geometric "laps" spaced at distance `w`.
2.  **Node Connectivity**:
    -   **Same-Lap Edges**: Vertical/Horizontal connections along the same sweep line.
    -   **Cross-Lap Edges**: Lateral connections to adjacent laps, limited by distance (typically $\sqrt{2} \cdot w$).
3.  **Connectivity Repair**: If SLAM updates create gaps or "islands" in the graph, the `repair_connectivity` algorithm uses BFS to find collision-free "bridges" between disjoint components.

### Graph Pruning (Essential Nodes)
To keep the graph efficient, HazMap uses a pruning algorithm that only keeps "essential" nodes:
-   **Terminal Nodes**: Nodes at the start or end of a lap.
-   **Boundary Nodes**: Nodes adjacent to unknown space.
-   **Connectivity Hubs**: Nodes that provide unique cross-lap connectivity to adjacent areas.

---

## 3. YOLO Visual Detection Pipeline

The visual pipeline provides 3D localization for objects of interest (fire, spills, smoke) using a YOLOv8 detector.

### 3.D Localization Logic
1.  **RGB-D Synchronization**: Uses `ApproximateTimeSynchronizer` to pair color frames with depth frames.
2.  **YOLO Inference**: Runs detection on the RGB frame to get bounding boxes and confidence scores.
3.  **3D Projection**: The center pixel of the bounding box is paired with the corresponding depth value. Using camera intrinsics ($f_x, f_y, c_x, c_y$), the system projects the 2D point into 3D camera coordinates:
    -   $X_{cam} = (u - c_x) \cdot Z / f_x$
    -   $Y_{cam} = (v - c_y) \cdot Z / f_y$
4.  **TF2 Transformation**: The point is then transformed from the `camera_color_optical_frame` to the `map` frame using ROS 2 TF2 frames.

### Object Tracking
-   **Exponential Moving Average (EMA)**: Positions of detected objects are smoothed over time to reduce jitter.
-   **Duplicate Suppression**: New detections are matched against existing objects within a `DISTANCE_THRESHOLD`.

---

## 4. Consolidated Hazard Impact Map

At the end of a mission, HazMap fuses all multi-modal data into a single, publication-ready **Hazard Impact Map**.

### Fusion Algorithm
The final impact value for any grid cell $(r, c)$ is calculated as a weighted average:

$$Impact(r, c) = \frac{\sum (Weight_i \cdot NormalizedGrid_i(r, c))}{\sum Weight_i}$$

-   **Sensor Weight**: $1 / Priority$ (e.g., CO with Priority 1 has more weight than Methane with Priority 3).
-   **Detection Weight**: Detections are converted into a "Gaussian Splash Impact Grid" where the intensity is based on the detection count and severity.
-   **Topographic Contours**: The map includes contour lines at 50%, 70%, and 90% levels to clearly demarcate high-risk zones.

---

## 5. Result Extraction

The system automatically generates a timestamped results directory containing:
-   **Individual Heatmaps**: Per-sensor Z-score maps.
-   **Detection Pins**: A map showing confirmed YOLO detections with splash radii.
-   **Connectivity Graph**: A visualization of the RCG with the robot's actual executed path overlaid.
-   **Consolidated Map**: The final fused hazard assessment.
