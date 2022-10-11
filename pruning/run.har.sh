#!/bin/sh
echo "[HAR]-R=0.3- Prune_energy"

bash prune.har.I_0.sh > new_data/[HAR]-R3-Prune_intermittent_0_log.txt
echo "[HAR]-R3- Prune_intermittent_0"
bash prune.har.E_0.sh > new_data/[HAR]-R3-Prune_energy_0_log.txt
echo "[HAR]-R3- Prune_energy_0"

bash prune.har.I_1.sh > new_data/[HAR]-R3-Prune_intermittent_1_log.txt
echo "[HAR]-R3- Prune_intermittent_1"
bash prune.har.E_1.sh > new_data/[HAR]-R3-Prune_energy_1_log.txt
echo "[HAR]-R3- Prune_energy_1"

bash prune.har.I_2.sh > new_data/[HAR]-R3-Prune_intermittent_2_log.txt
echo "[HAR]-R3- Prune_intermittent_2"
bash prune.har.E_2.sh > new_data/[HAR]-R3-Prune_energy_2_log.txt
echo "[HAR]-R3- Prune_energy_2"

mkdir har-test-R3
mkdir har-test-R3/I
mkdir har-test-R3/E

cp logs/intermittent/HAR/* har-test-R3/I/
cp logs/energy/HAR/* har-test-R3/E/
cp saved_models/HAR.origin.pth.tar har-test-R3/
cp saved_models/intermittent/HAR/* har-test-R3/I/
cp saved_models/energy/HAR/* har-test-R3/E/






