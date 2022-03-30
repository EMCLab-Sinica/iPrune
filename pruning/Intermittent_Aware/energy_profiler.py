import pandas as pd

file1 = '../../msp430/.energytrace/EnergyTrace_2022_03_29_203612.csv'
file2 = '../../msp430/.energytrace/energy_prune.csv'
file3 = '../../msp430/.energytrace/unprune.csv'
file4 = '../../msp430/.energytrace/1.csv'
f = pd.read_csv(file4, sep=',')

start = 10000000000000000000000000000000000
finish = -1
start_idx = -1
finish_idx = -1
for i in range(len(f)):
    mW = f['Current (nA)'][i] * f['Voltage (mV)'][i] / 10 ** 9
    time = f['Time (ns)'][i]
    if 22.5 > mW > 19.9 and time > 9 * 10 ** 9 and time < 22 * 10 ** 9:
        if start > time:
            start = time
            start_idx = i
        if finish < time:
            finish = time
            finish_idx = i

print(start, finish)
print(f['Energy (uJ)'][start_idx], f['Energy (uJ)'][finish_idx])
print((f['Energy (uJ)'][finish_idx] - f['Energy (uJ)'][start_idx]) / 10 ** 4)
