#!/usr/bin/env python3
"""
MQ-series analog gas sensor reader using ADS1115 over I2C.

Publishes:
- gas/voltage (Float32): sensor output voltage (V)
- gas/ppm (Float32): estimated concentration using curve ppm = A * (Rs/R0)^(-B)

Parameters:
- i2c_addr (int, default 0x48)
- i2c_bus (int, default 1)
- channel (int 0..3, default 0)
- vref (float, default 5.0)  # sensor supply used in divider
- rl (float, default 10000.0)  # load resistor (Ohm)
- r0 (float, default 10000.0)  # sensor resistance in clean air (Ohm)
- A (float, default 1000.0)
- B (float, default 1.5)
- rate_hz (float, default 10.0)

Note: Calibrate R0, A, and B per the specific MQ sensor and gas.
"""

import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

try:
    from smbus2 import SMBus  # type: ignore
except Exception:  # pragma: no cover - environment without I2C
    SMBus = None  # type: ignore


# ADS1115 register map and constants
ADS1115_ADDR = 0x48
REG_CONV = 0x00
REG_CONF = 0x01

# MUX values for single-ended AINx relative to GND
MUX_BY_CH = {0: 0x4000, 1: 0x5000, 2: 0x6000, 3: 0x7000}

# PGA ±4.096V (LSB = 125 uV)
PGA_4_096V = 0x0200
MODE_SINGLE = 0x0100
DR_1600SPS = 0x0080
OS_START = 0x8000


def write16(bus, addr, reg, value):
    bus.write_i2c_block_data(addr, reg, [(value >> 8) & 0xFF, value & 0xFF])


def read16(bus, addr, reg):
    data = bus.read_i2c_block_data(addr, reg, 2)
    val = (data[0] << 8) | data[1]
    if val & 0x8000:
        val -= 1 << 16
    return val


class MQADS1115Node(Node):
    def __init__(self) -> None:
        super().__init__('mq_ads1115_node')

        self.addr = (
            self.declare_parameter('i2c_addr', ADS1115_ADDR)
            .get_parameter_value()
            .integer_value
        )
        self.bus_id = (
            self.declare_parameter('i2c_bus', 1).get_parameter_value().integer_value
        )
        self.channel = (
            self.declare_parameter('channel', 0).get_parameter_value().integer_value
        )
        self.vref = (
            self.declare_parameter('vref', 5.0).get_parameter_value().double_value
        )
        self.rl = (
            self.declare_parameter('rl', 10_000.0).get_parameter_value().double_value
        )
        self.r0 = (
            self.declare_parameter('r0', 10_000.0).get_parameter_value().double_value
        )
        self.A = (
            self.declare_parameter('A', 1000.0).get_parameter_value().double_value
        )
        self.B = (
            self.declare_parameter('B', 1.5).get_parameter_value().double_value
        )
        self.rate_hz = (
            self.declare_parameter('rate_hz', 10.0).get_parameter_value().double_value
        )

        if SMBus is None:
            self.get_logger().error(
                'smbus2 not available; install with "pip install smbus2" or apt install python3-smbus'
            )
            raise RuntimeError('smbus2 not available')

        self.bus = SMBus(self.bus_id)

        self.pub_v = self.create_publisher(Float32, 'gas/voltage', 10)
        self.pub_ppm = self.create_publisher(Float32, 'gas/ppm', 10)

        self.timer = self.create_timer(max(0.01, 1.0 / self.rate_hz), self._tick)
        self.get_logger().info(
            f"MQ ADS1115 reader started: addr=0x{self.addr:02X} bus={self.bus_id} ch={self.channel}"
        )

    def _tick(self) -> None:
        try:
            mux = MUX_BY_CH.get(int(self.channel), 0x4000)
            conf = OS_START | mux | PGA_4_096V | MODE_SINGLE | DR_1600SPS | 0x0003
            write16(self.bus, self.addr, REG_CONF, conf)
            # wait for conversion (1/1600s) plus small margin
            time.sleep(1.0 / 1600.0 + 0.001)
            raw = read16(self.bus, self.addr, REG_CONV)
            volts = raw * 0.000125  # 125 uV per LSB at ±4.096V full-scale

            # Divider: RL to GND, Rs to Vref, Vout between Rs and RL
            vout = float(max(1e-6, min(self.vref - 1e-6, volts)))
            rs = self.rl * (self.vref / vout - 1.0)
            ratio = rs / self.r0 if self.r0 > 0 else 0.0
            ppm = float(self.A * (ratio ** (-self.B)))

            mv = Float32()
            mv.data = vout
            self.pub_v.publish(mv)

            mp = Float32()
            mp.data = ppm
            self.pub_ppm.publish(mp)
        except Exception as e:
            self.get_logger().warn(f'I2C read error: {e}')


def main() -> None:
    rclpy.init()
    node = MQADS1115Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
