import argparse
import logging
import time
from typing import Optional

import pyvisa

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger('control-power-supply')

class DeviceNotFound(Exception):
    pass

class Device:
    BAUD_RATE: Optional[int] = None
    IDENTIFIER: str

    def __init__(self):
        rm = pyvisa.ResourceManager()
        resources = rm.list_resources()
        print(resources)

        try:
            self.inst = rm.open_resource(next(
                resource for resource in resources if self.IDENTIFIER in resource))
        except StopIteration:
            raise DeviceNotFound

        if self.BAUD_RATE:
            self.inst.baud_rate = self.BAUD_RATE
        print(self.inst.query('*IDN?'))

    def _write_command(self, command: str):
        logger.debug('Sending command %s', command)
        self.inst.write(command)

    def output_on(self):
        raise NotImplementedError

    def output_off(self):
        raise NotImplementedError

    def set_voltage(self, voltage):
        raise NotImplementedError

class BK9171B(Device):
    BAUD_RATE = 57600  # value from BK Precision's software
    IDENTIFIER = 'ttyUSB'
    #IDENTIFIER = 'COM3'

    def output_on(self):
        self._write_command('OUT 1')

    def output_off(self):
        self._write_command('OUT 0')

    def set_voltage(self, voltage):
        self._write_command(f'VSET {voltage}')

class SCPIDevice(Device):
    # Implements SCPI (Standard Commands for Programmable Instruments)
    # Ref: https://www.ivifoundation.org/docs/scpi-99.pdf

    def output_on(self):
        # 15.12
        self._write_command('OUTPut:STATe ON')

    def output_off(self):
        # 15.12
        self._write_command('OUTPut:STATe OFF')

    def set_voltage(self, voltage):
        # 19.23.4.1.1
        self._write_command(f'SOURce:VOLTage:LEVel:IMMediate:AMPLitude {voltage}')

class Keithley2280S(SCPIDevice):
    IDENTIFIER = '1510::8832'  # decimal of its USB ID 05e6:2280

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--interval_on', type=float, required=True)
    parser.add_argument('--interval_off', type=float, required=True)
    parser.add_argument('--v_high', type=float, default=2.8)
    parser.add_argument('--v_low', type=float, default=0.5)
    args = parser.parse_args()

    start_time = time.time()
    counter = 0

    try:
        device = Keithley2280S()
    except DeviceNotFound:
        device = BK9171B()

    try:
        device.output_on()
        while True:
            device.set_voltage(args.v_high)
            print(time.time() - start_time)
            time.sleep(args.interval_on)
            device.set_voltage(args.v_low)
            print(time.time() - start_time)
            time.sleep(args.interval_off)
            counter += 1
    except:
        device.output_off()
        device.set_voltage(args.v_high)
        time.sleep(1)
        device.output_on()
        print(f'power cycles = {counter/2}')

if __name__ == '__main__':
    main()
