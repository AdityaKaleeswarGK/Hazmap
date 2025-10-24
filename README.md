The inital config of Jetacker ...

## Gas mapping (MQ sensors + ADS1115)

This workspace includes a 4-channel MQ gas sensor reader (via ADS1115) and a gas mapper that produces separate heatmaps per gas type.

Topics published (defaults):
- gas/methane/ppm and gas/methane/voltage (MQ-4)
- gas/lpg/ppm and gas/lpg/voltage (MQ-5)
- gas/co/ppm and gas/co/voltage (MQ-7)
- gas/air_quality/ppm and gas/air_quality/voltage (MQ-135)
- gas_map/<name> OccupancyGrid per gas

Run on a Jetson device with ROS 2:
```bash
source /opt/ros/$ROS_DISTRO/setup.bash
colcon build --symlink-install
source install/setup.bash
export need_compile=True
ros2 launch app gas_mapping.launch.py rate_hz:=10.0
```

Threshold for plotting on maps:
- Edit `PPM_THRESHOLD_MIN` at the top of `src/app/app/gas_mapper.py` (default 20 ppm), or pass `ppm_threshold` as a parameter in the launch file.

Map scaling to 0..100 uses per-gas max ppm defaults based on typical sensor ranges (MQ-4/5/7: 10000, MQ-135 air_quality: 1000). Adjust `topic_max_ppm` in `app/launch/gas_mapping.launch.py` or via ROS parameters as needed.

