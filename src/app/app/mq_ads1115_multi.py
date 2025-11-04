#!/usr/bin/env python3
"""
Read multiple MQ sensors via one ADS1115 at equal intervals and publish per-gas topics.

Parameters:
- i2c_addr: 0x48
- i2c_bus: 1
- rate_hz: 10.0  (all sensors read each cycle; equal cadence)
- sensors: list of dicts, each with:
    name: 'methane' | 'lpg' | 'co' | 'air_quality'
    channel: 0|1|2|3
    vref: 5.0
    rl: 10000.0
    r0: 12000.0
    A: 1000.0
    B: 1.5

Publishes for each sensor name N:
- gas/N/voltage (std_msgs/Float32)
- gas/N/ppm (std_msgs/Float32)
"""

import time
from typing import List, Dict, Any

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

try:
    from smbus2 import SMBus  # type: ignore
except Exception:  # pragma: no cover
    SMBus = None  # type: ignore


ADS1115_ADDR = 0x48
REG_CONV = 0x00
REG_CONF = 0x01
MUX_BY_CH = {0: 0x4000, 1: 0x5000, 2: 0x6000, 3: 0x7000}
PGA_4_096V = 0x0200
MODE_SINGLE = 0x0100
DR_860SPS = 0x00E0  # slightly slower but robust
OS_START = 0x8000


def write16(bus, addr, reg, value):
    bus.write_i2c_block_data(addr, reg, [(value >> 8) & 0xFF, value & 0xFF])


def read16(bus, addr, reg):
    data = bus.read_i2c_block_data(addr, reg, 2)
    val = (data[0] << 8) | data[1]
    if val & 0x8000:
        val -= 1 << 16
    return val


class ADS1115Multi(Node):
    def __init__(self) -> None:
        super().__init__('mq_ads1115_multi')

        self.addr = (
            self.declare_parameter('i2c_addr', ADS1115_ADDR).get_parameter_value().integer_value
        )
        self.bus_id = (
            self.declare_parameter('i2c_bus', 1).get_parameter_value().integer_value
        )
        self.rate_hz = (
            self.declare_parameter('rate_hz', 10.0).get_parameter_value().double_value
        )

        raw_sensors = self.declare_parameter('sensors', [
            {'name': 'methane', 'channel': 0, 'vref': 5.0, 'rl': 10_000.0, 'r0': 12_000.0, 'A': 1000.0, 'B': 1.5},
            {'name': 'lpg', 'channel': 1, 'vref': 5.0, 'rl': 10_000.0, 'r0': 12_000.0, 'A': 1000.0, 'B': 1.5},
            {'name': 'co', 'channel': 2, 'vref': 5.0, 'rl': 10_000.0, 'r0': 12_000.0, 'A': 1000.0, 'B': 1.5},
            {'name': 'air_quality', 'channel': 3, 'vref': 5.0, 'rl': 10_000.0, 'r0': 12_000.0, 'A': 1000.0, 'B': 1.5},
        ]).value

        self.sensors: List[Dict[str, Any]] = self._parse_list(raw_sensors)
        if SMBus is None:
            self.get_logger().error('smbus2 not available; install smbus2 / python3-smbus')
            raise RuntimeError('smbus2 not available')

        self.bus = SMBus(self.bus_id)
        # Create publishers per sensor
        self.pubs_v = {}
        self.pubs_ppm = {}
        for s in self.sensors:
            name = str(s.get('name', 'ch'))
            self.pubs_v[name] = self.create_publisher(Float32, f'gas/{name}/voltage', 10)
            self.pubs_ppm[name] = self.create_publisher(Float32, f'gas/{name}/ppm', 10)

        period = max(0.01, 1.0 / max(0.1, self.rate_hz))
        self.timer = self.create_timer(period, self._tick)
        self.get_logger().info(f'ADS1115 multi started addr=0x{self.addr:02X} bus={self.bus_id} rate={self.rate_hz}Hz with {len(self.sensors)} sensors')

    def _parse_list(self, raw):
        if isinstance(raw, list):
            # already list of dicts or list of yaml strings
            if raw and isinstance(raw[0], str):
                try:
                    import yaml  # type: ignore
                    return yaml.safe_load('\n'.join(raw))
                except Exception:
                    return []
            return raw
        if isinstance(raw, str):
            try:
                import yaml  # type: ignore
                return yaml.safe_load(raw)
            except Exception:
                return []
        return []

    def _read_channel_volts(self, ch: int) -> float:
        mux = MUX_BY_CH.get(int(ch), 0x4000)
        conf = OS_START | mux | PGA_4_096V | MODE_SINGLE | DR_860SPS | 0x0003
        write16(self.bus, self.addr, REG_CONF, conf)
        # 860 sps -> ~1.16ms per sample; add margin
        time.sleep(0.002)
        raw = read16(self.bus, self.addr, REG_CONV)
        volts = raw * 0.000125  # 125 uV per LSB
        return volts

    def _tick(self) -> None:
        for s in self.sensors:
            try:
                name = str(s.get('name', 'ch'))
                ch = int(s.get('channel', 0))
                vref = float(s.get('vref', 5.0))
                rl = float(s.get('rl', 10_000.0))
                r0 = float(s.get('r0', 10_000.0))
                A = float(s.get('A', 1000.0))
                B = float(s.get('B', 1.5))

                volts = self._read_channel_volts(ch)
                vout = float(max(1e-6, min(vref - 1e-6, volts)))
                rs = rl * (vref / vout - 1.0)
                ratio = rs / r0 if r0 > 0 else 0.0
                ppm = float(A * (ratio ** (-B)))

                mv = Float32(); mv.data = vout
                self.pubs_v[name].publish(mv)
                mp = Float32(); mp.data = ppm
                self.pubs_ppm[name].publish(mp)
            except Exception as e:
                self.get_logger().warn(f'sensor {s} read error: {e}')


def main() -> None:
    rclpy.init()
    node = ADS1115Multi()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
