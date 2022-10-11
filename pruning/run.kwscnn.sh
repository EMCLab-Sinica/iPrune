#!/bin/sh

echo "[KWS_CNN]-R=0.1"

bash prune.kws_cnn_s-R1.I_0.sh > new_data/[KWS_CNN]-R1-Prune_intermittent_0_log.txt
echo "[KWS_CNN]-R1- Prune_intermittent_0"
bash prune.kws_cnn_s-R1.E_0.sh > new_data/[KWS_CNN]-R1-Prune_energy_0_log.txt
echo "[KWS_CNN]-R1- Prune_energy_0"

bash prune.kws_cnn_s-R1.I_1.sh > new_data/[KWS_CNN]-R1-Prune_intermittent_1_log.txt
echo "[KWS_CNN]-R1- Prune_intermittent_1"
bash prune.kws_cnn_s-R1.E_1.sh > new_data/[KWS_CNN]-R1-Prune_energy_1_log.txt
echo "[KWS_CNN]-R1- Prune_energy_1"

bash prune.kws_cnn_s-R1.I_2.sh > new_data/[KWS_CNN]-R1-Prune_intermittent_2_log.txt
echo "[KWS_CNN]-R1- Prune_intermittent_2"
bash prune.kws_cnn_s-R1.E_2.sh > new_data/[KWS_CNN]-R1-Prune_energy_2_log.txt
echo "[KWS_CNN]-R1- Prune_energy_2"

bash prune.kws_cnn_s-R1.I_3.sh > new_data/[KWS_CNN]-R1-Prune_intermittent_3_log.txt
echo "[KWS_CNN]-R1- Prune_intermittent_3"
bash prune.kws_cnn_s-R1.E_3.sh > new_data/[KWS_CNN]-R1-Prune_energy_3_log.txt
echo "[KWS_CNN]-R1- Prune_energy_3"

bash prune.kws_cnn_s-R1.I_4.sh > new_data/[KWS_CNN]-R1-Prune_intermittent_4_log.txt
echo "[KWS_CNN]-R1- Prune_intermittent_4"
bash prune.kws_cnn_s-R1.E_4.sh > new_data/[KWS_CNN]-R1-Prune_energy_4_log.txt
echo "[KWS_CNN]-R1- Prune_energy_4"

bash prune.kws_cnn_s-R1.I_5.sh > new_data/[KWS_CNN]-R1-Prune_intermittent_5_log.txt
echo "[KWS_CNN]-R1- Prune_intermittent_5"
bash prune.kws_cnn_s-R1.E_5.sh > new_data/[KWS_CNN]-R1-Prune_energy_5_log.txt
echo "[KWS_CNN]-R1- Prune_energy_5"

bash prune.kws_cnn_s-R1.I_6.sh > new_data/[KWS_CNN]-R1-Prune_intermittent_6_log.txt
echo "[KWS_CNN]-R1- Prune_intermittent_6"
bash prune.kws_cnn_s-R1.E_6.sh > new_data/[KWS_CNN]-R1-Prune_energy_6_log.txt
echo "[KWS_CNN]-R1- Prune_energy_6"

mkdir KWS_CNN_S-test-R1
mkdir KWS_CNN_S-test-R1/I
mkdir KWS_CNN_S-test-R1/E


cp logs/intermittent/KWS_CNN_S/* KWS_CNN_S-test-R1/I/
cp logs/energy/KWS_CNN_S/* KWS_CNN_S-test-R1/E/
cp saved_models/KWS_CNN_S.origin.pth.tar KWS_CNN_S-test-R1/
cp saved_models/intermittent/KWS_CNN_S/* KWS_CNN_S-test-R1/I/
cp saved_models/energy/KWS_CNN_S/* KWS_CNN_S-test-R1/E/


echo "[KWS_CNN]-R=0.3"

bash prune.kws_cnn_s-R3.I_0.sh > new_data/[KWS_CNN]-R3-Prune_intermittent_0_log.txt
echo "[KWS_CNN]-R3- Prune_intermittent_0"
bash prune.kws_cnn_s-R3.E_0.sh > new_data/[KWS_CNN]-R3-Prune_energy_0_log.txt
echo "[KWS_CNN]-R3- Prune_energy_0"

bash prune.kws_cnn_s-R3.I_1.sh > new_data/[KWS_CNN]-R3-Prune_intermittent_1_log.txt
echo "[KWS_CNN]-R3- Prune_intermittent_1"
bash prune.kws_cnn_s-R3.E_1.sh > new_data/[KWS_CNN]-R3-Prune_energy_1_log.txt
echo "[KWS_CNN]-R3- Prune_energy_1"

bash prune.kws_cnn_s-R3.I_2.sh > new_data/[KWS_CNN]-R3-Prune_intermittent_2_log.txt
echo "[KWS_CNN]-R3- Prune_intermittent_2"
bash prune.kws_cnn_s-R3.E_2.sh > new_data/[KWS_CNN]-R3-Prune_energy_2_log.txt
echo "[KWS_CNN]-R3- Prune_energy_2"

bash prune.kws_cnn_s-R3.I_3.sh > new_data/[KWS_CNN]-R3-Prune_intermittent_3_log.txt
echo "[KWS_CNN]-R3- Prune_intermittent_3"
bash prune.kws_cnn_s-R3.E_3.sh > new_data/[KWS_CNN]-R3-Prune_energy_3_log.txt
echo "[KWS_CNN]-R3- Prune_energy_3"

mkdir KWS_CNN_S-test-R3
mkdir KWS_CNN_S-test-R3/I
mkdir KWS_CNN_S-test-R3/E


cp logs/intermittent/KWS_CNN_S/* KWS_CNN_S-test-R3/I/
cp logs/energy/KWS_CNN_S/* KWS_CNN_S-test-R3/E/
cp saved_models/KWS_CNN_S.origin.pth.tar KWS_CNN_S-test-R3/
cp saved_models/intermittent/KWS_CNN_S/* KWS_CNN_S-test-R3/I/
cp saved_models/energy/KWS_CNN_S/* KWS_CNN_S-test-R3/E/


echo "[KWS_CNN]-R=0.2"

bash prune.kws_cnn_s-R2.I_0.sh > new_data/[KWS_CNN]-R2-Prune_intermittent_0_log.txt
echo "[KWS_CNN]-R2- Prune_intermittent_0"
bash prune.kws_cnn_s-R2.E_0.sh > new_data/[KWS_CNN]-R2-Prune_energy_0_log.txt
echo "[KWS_CNN]-R2- Prune_energy_0"

bash prune.kws_cnn_s-R2.I_1.sh > new_data/[KWS_CNN]-R2-Prune_intermittent_1_log.txt
echo "[KWS_CNN]-R2- Prune_intermittent_1"
bash prune.kws_cnn_s-R2.E_1.sh > new_data/[KWS_CNN]-R2-Prune_energy_1_log.txt
echo "[KWS_CNN]-R2- Prune_energy_1"

bash prune.kws_cnn_s-R2.I_2.sh > new_data/[KWS_CNN]-R2-Prune_intermittent_2_log.txt
echo "[KWS_CNN]-R2- Prune_intermittent_2"
bash prune.kws_cnn_s-R2.E_2.sh > new_data/[KWS_CNN]-R2-Prune_energy_2_log.txt
echo "[KWS_CNN]-R2- Prune_energy_2"

bash prune.kws_cnn_s-R2.I_3.sh > new_data/[KWS_CNN]-R2-Prune_intermittent_3_log.txt
echo "[KWS_CNN]-R2- Prune_intermittent_3"
bash prune.kws_cnn_s-R2.E_3.sh > new_data/[KWS_CNN]-R2-Prune_energy_3_log.txt
echo "[KWS_CNN]-R2- Prune_energy_3"

mkdir KWS_CNN_S-test-R2
mkdir KWS_CNN_S-test-R2/I
mkdir KWS_CNN_S-test-R2/E


cp logs/intermittent/KWS_CNN_S/* KWS_CNN_S-test-R2/I/
cp logs/energy/KWS_CNN_S/* KWS_CNN_S-test-R2/E/
cp saved_models/KWS_CNN_S.origin.pth.tar KWS_CNN_S-test-R2/
cp saved_models/intermittent/KWS_CNN_S/* KWS_CNN_S-test-R2/I/
cp saved_models/energy/KWS_CNN_S/* KWS_CNN_S-test-R2/E/


