Place your Ultralytics YOLO .pt model files here.

Recommended file name: best.pt

At install time, files in this folder are copied to:
share/large_models/models

The yolo_infer node resolves the model path as:
- If the model_path ROS param is set and exists, it uses that;
- Else, it loads share/large_models/models/best.pt.

Example override:
ros2 run large_models yolo_infer --ros-args -p model_path:=/absolute/path/to/your.pt
