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
DR_128SPS = 0x0080  
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

        # NOTE: ROS 2 parameters cannot be lists of dictionaries. We therefore
        # expose the sensors configuration as a single JSON string parameter.
        # Users may override with either JSON or YAML text; we parse at runtime.
        default_sensors_json = (
            '[{"name":"methane","channel":0,"vref":5.0,"rl":10000.0,"r0":12000.0,"A":1000.0,"B":1.5},'
            ' {"name":"lpg","channel":1,"vref":5.0,"rl":10000.0,"r0":12000.0,"A":1000.0,"B":1.5},'
            ' {"name":"co","channel":2,"vref":5.0,"rl":10000.0,"r0":12000.0,"A":1000.0,"B":1.5},'
            ' {"name":"air_quality","channel":3,"vref":5.0,"rl":10000.0,"r0":12000.0,"A":1000.0,"B":1.5}]'
        )
        self.declare_parameter('sensors', default_sensors_json)
        raw_sensors = self.get_parameter('sensors').get_parameter_value().string_value
        self.sensors: List[Dict[str, Any]] = self._parse_sensors(raw_sensors, default_sensors_json)
        if SMBus is None:
            self.get_logger().error('smbus2 not available; install smbus2 / python3-smbus')
            raise RuntimeError('smbus2 not available')

        self.simulate = self.declare_parameter('simulate', False).get_parameter_value().bool_value
        self.max_retries = int(self.declare_parameter('i2c_retries', 3).get_parameter_value().integer_value)
        self.pga_range_v = float(self.declare_parameter('pga_range_v', 4.096).get_parameter_value().double_value)
        # ADS1115 LSB size depends on chosen PGA range: LSB = range/32768
        self.lsb_volts = self.pga_range_v / 32768.0

        self.bus = SMBus(self.bus_id) if not self.simulate else None

        if not self.simulate:
            self._i2c_scan_once()
            if not self._device_present(self.addr):
                self.get_logger().warn(f'ADS1115 address 0x{self.addr:02X} not detected on i2c-{self.bus_id}; proceeding but reads will fail (Errno 121).')
        else:
            self.get_logger().warn('Simulation mode enabled: generating synthetic voltages.')
        self.pubs_v = {}
        self.pubs_ppm = {}
        for s in self.sensors:
            name = str(s.get('name', 'ch'))
            self.pubs_v[name] = self.create_publisher(Float32, f'gas/{name}/voltage', 10)
            self.pubs_ppm[name] = self.create_publisher(Float32, f'gas/{name}/ppm', 10)

        period = max(0.01, 1.0 / max(0.1, self.rate_hz))
        self.timer = self.create_timer(period, self._tick)
        self.get_logger().info(f'ADS1115 multi started addr=0x{self.addr:02X} bus={self.bus_id} rate={self.rate_hz}Hz with {len(self.sensors)} sensors')

    def _parse_sensors(self, raw: str, fallback: str) -> List[Dict[str, Any]]:
        """Parse sensors definition from a JSON or YAML string.

        Returns a list of sensor dicts. On any failure returns the fallback
        (default) sensors definition.
        """
        text = raw.strip() or fallback
        # Try JSON first
        import json
        try:
            val = json.loads(text)
            if isinstance(val, list):
                return [v for v in val if isinstance(v, dict)]
        except Exception:
            pass
        # Try YAML if available
        try:
            import yaml  # type: ignore
            val = yaml.safe_load(text)
            if isinstance(val, list):
                return [v for v in val if isinstance(v, dict)]
        except Exception:
            pass
        # Fallback to default JSON
        try:
            val = json.loads(fallback)
            return [v for v in val if isinstance(v, dict)]
        except Exception:
            return []

    def _read_channel_volts(self, ch: int) -> float:
        if self.simulate:
            # Deterministic synthetic signal per channel
            return 1.0 + 0.25 * ((time.time()/3.0 + ch) % 1.0)
        mux = MUX_BY_CH.get(int(ch), 0x4000)
        # Build configuration word: start single conversion, selected mux, PGA per pga_range_v, single-shot mode, data rate 128SPS
        pga_bits = PGA_4_096V  # keep constant; scaling uses self.lsb_volts
        conf = OS_START | mux | pga_bits | MODE_SINGLE | DR_128SPS | 0x0003
        for attempt in range(self.max_retries):
            try:
                write16(self.bus, self.addr, REG_CONF, conf)
                # Wait until conversion complete (poll OS bit) with timeout ~10ms
                for _ in range(10):
                    raw_conf = read16(self.bus, self.addr, REG_CONF)
                    if raw_conf & OS_START:  # OS bit returns to 1 when conversion finished
                        break
                    time.sleep(0.001)
                raw = read16(self.bus, self.addr, REG_CONV)
                volts = raw * self.lsb_volts
                return volts
            except Exception as e:
                if attempt + 1 == self.max_retries:
                    raise
                time.sleep(0.002)

    def _i2c_scan_once(self):
        try:
            found = []
            for address in range(0x03, 0x77):
                try:
                    self.bus.write_quick(address)
                    found.append(address)
                except Exception:
                    pass
            if found:
                hex_list = ','.join(f'0x{a:02X}' for a in found)
                self.get_logger().info(f'I2C scan bus {self.bus_id}: found {len(found)} device(s): {hex_list}')
            else:
                self.get_logger().warn(f'I2C scan bus {self.bus_id}: no devices detected')
        except Exception as e:
            self.get_logger().warn(f'I2C scan failed: {e}')

    def _device_present(self, address: int) -> bool:
        try:
            self.bus.write_quick(address)
            return True
        except Exception:
            return False

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

                try:
                    volts = self._read_channel_volts(ch)
                except Exception as e:
                    self.get_logger().warn(f'channel {ch} read failed after {self.max_retries} retries: {e}')
                    continue
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
