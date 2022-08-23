import csv
import pyvisa
import numpy as np
from struct import unpack

rm = pyvisa.ResourceManager()
tek = None
for resource in rm.list_resources():
    parts = resource.split('::')
    if not parts[0].startswith('USB'):
        continue
    if int(parts[1]) == 0x0699:  # USB vendor ID: Tektronix
        tek = resource
        break

if not tek:
    raise RuntimeError('Cannot find a Tektronix oscilloscope')

scope = rm.open_resource(tek)

print(scope.query("*IDN?"))

# Modified from https://forum.tek.com/viewtopic.php?t=137002

def read_curve(channel):
    scope.write(f'DATA:SOU {channel}')
    scope.write('DATA:WIDTH 1')
    scope.write('DATA:ENC RPB')

    ymult = float(scope.query('WFMPRE:YMULT?'))
    yzero = float(scope.query('WFMPRE:YZERO?'))
    yoff = float(scope.query('WFMPRE:YOFF?'))
    xincr = float(scope.query('WFMPRE:XINCR?'))

    print((ymult, yzero, yoff, xincr))
    scope.write('CURVE?')
    data = scope.read_raw()
    headerlen = 2 + int(data[1])
    ADC_wave = data[headerlen:-1]

    ADC_wave = np.array(unpack('%sB' % len(ADC_wave),ADC_wave))

    volts = (ADC_wave - yoff) * ymult  + yzero

    times = np.arange(0, xincr * len(volts), xincr)

    return times, volts

with open('tek.csv', 'x', newline='') as csvfile:
    times_ch1, volts_ch1 = read_curve('CH1')
    times_ch2, volts_ch2 = read_curve('CH2')

    writer = csv.writer(csvfile)

    for time_ch1, time_ch2, volt_ch1, volt_ch2 in zip(times_ch1, times_ch2, volts_ch1, volts_ch2):
        assert time_ch1 == time_ch2
        writer.writerow([time_ch1, volt_ch1, volt_ch2])
