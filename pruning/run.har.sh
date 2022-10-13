#!/bin/sh
echo "[HAR]-R=0.3"

bash prune.har_R3.I_0.sh > new_data/[HAR]-R3-Prune_intermittent_0_log.txt
echo "[HAR]-R3- Prune_intermittent_0"
bash prune.har_R3.E_0.sh > new_data/[HAR]-R3-Prune_energy_0_log.txt
echo "[HAR]-R3- Prune_energy_0"

bash prune.har_R3.I_1.sh > new_data/[HAR]-R3-Prune_intermittent_1_log.txt
echo "[HAR]-R3- Prune_intermittent_1"
bash prune.har_R3.E_1.sh > new_data/[HAR]-R3-Prune_energy_1_log.txt
echo "[HAR]-R3- Prune_energy_1"
 
bash prune.har_R3.I_2.sh > new_data/[HAR]-R3-Prune_intermittent_2_log.txt
echo "[HAR]-R3- Prune_intermittent_2"
bash prune.har_R3.E_2.sh > new_data/[HAR]-R3-Prune_energy_2_log.txt
echo "[HAR]-R3- Prune_energy_2"

bash prune.har_R3.I_3.sh > new_data/[HAR]-R3-Prune_intermittent_3_log.txt
echo "[HAR]-R3- Prune_intermittent_3"
bash prune.har_R3.E_3.sh > new_data/[HAR]-R3-Prune_energy_3_log.txt
echo "[HAR]-R3- Prune_energy_3"

cp logs/intermittent/HAR/* har-test-R3/I/
cp logs/energy/HAR/* har-test-R3/E/
cp saved_models/intermittent/HAR/* har-test-R3/I/
cp saved_models/energy/HAR/* har-test-R3/E/


echo "[HAR]-R=0.1"


bash prune.har_R1.I_0.sh > new_data/[HAR]-R1-Prune_intermittent_0_log.txt
echo "[HAR]-R1- Prune_intermittent_0"
bash prune.har_R1.E_0.sh > new_data/[HAR]-R1-Prune_energy_0_log.txt
echo "[HAR]-R1- Prune_energy_0"

bash prune.har_R1.I_1.sh > new_data/[HAR]-R1-Prune_intermittent_1_log.txt
echo "[HAR]-R1- Prune_intermittent_1"
bash prune.har_R1.E_1.sh > new_data/[HAR]-R1-Prune_energy_1_log.txt
echo "[HAR]-R1- Prune_energy_1"

bash prune.har_R1.I_2.sh > new_data/[HAR]-R1-Prune_intermittent_2_log.txt
echo "[HAR]-R1- Prune_intermittent_2"
bash prune.har_R1.E_2.sh > new_data/[HAR]-R1-Prune_energy_2_log.txt
echo "[HAR]-R1- Prune_energy_2"

bash prune.har_R1.I_3.sh > new_data/[HAR]-R1-Prune_intermittent_3_log.txt
echo "[HAR]-R1- Prune_intermittent_3"
bash prune.har_R1.E_3.sh > new_data/[HAR]-R1-Prune_energy_3_log.txt
echo "[HAR]-R1- Prune_energy_3"

bash prune.har_R1.I_4.sh > new_data/[HAR]-R1-Prune_intermittent_4_log.txt
echo "[HAR]-R1- Prune_intermittent_4"
bash prune.har_R1.E_4.sh > new_data/[HAR]-R1-Prune_energy_4_log.txt
echo "[HAR]-R1- Prune_energy_4"

bash prune.har_R1.I_5.sh > new_data/[HAR]-R1-Prune_intermittent_5_log.txt
echo "[HAR]-R1- Prune_intermittent_5"
bash prune.har_R1.E_5.sh > new_data/[HAR]-R1-Prune_energy_5_log.txt
echo "[HAR]-R1- Prune_energy_5"

bash prune.har_R1.I_6.sh > new_data/[HAR]-R1-Prune_intermittent_6_log.txt
echo "[HAR]-R1- Prune_intermittent_6"
bash prune.har_R1.E_6.sh > new_data/[HAR]-R1-Prune_energy_6_log.txt
echo "[HAR]-R1- Prune_energy_6"

mkdir har-test-R1
mkdir har-test-R1/I
mkdir har-test-R1/E


cp logs/intermittent/HAR/* har-test-R1/I/
cp logs/energy/HAR/* har-test-R1/E/
cp saved_models/HAR.origin.pth.tar har-test-R1/
cp saved_models/intermittent/HAR/* har-test-R1/I/
cp saved_models/energy/HAR/* har-test-R1/E/
