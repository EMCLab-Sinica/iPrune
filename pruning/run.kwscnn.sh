#!/bin/sh
echo "Start KWS_CNN model-2"

bash prune.kws_cnn_s.sh > new_data/[KWS_CNN]-2-Pretrained_log.txt
echo "[KWS_CNN]-2- Pretrained model"

bash prune.kws_cnn_s.I_0.sh > new_data/[KWS_CNN]-2-Prune_intermittent_0_log.txt
echo "[KWS_CNN]-2- Prune_intermittent_0"
bash prune.kws_cnn_s.E_0.sh > new_data/[KWS_CNN]-2-Prune_energy_0_log.txt
echo "[KWS_CNN]-2- Prune_energy_0"

bash prune.kws_cnn_s.I_1.sh > new_data/[KWS_CNN]-2-Prune_intermittent_1_log.txt
echo "[KWS_CNN]-2- Prune_intermittent_1"
bash prune.kws_cnn_s.E_1.sh > new_data/[KWS_CNN]-2-Prune_energy_1_log.txt
echo "[KWS_CNN]-2- Prune_energy_1"

bash prune.kws_cnn_s.I_2.sh > new_data/[KWS_CNN]-2-Prune_intermittent_2_log.txt
echo "[KWS_CNN]-2- Prune_intermittent_2"
bash prune.kws_cnn_s.E_2.sh > new_data/[KWS_CNN]-2-Prune_energy_2_log.txt
echo "[KWS_CNN]-2- Prune_energy_2"

mkdir kws-test-2
mkdir kws-test-2/I
mkdir kws-test-2/E

cp saved_models/KWS_CNN_S.origin.pth.tar kws-test-2/
cp saved_models/intermittent/KWS_CNN_S/* kws-test-2/I/
cp saved_models/energy/KWS_CNN_S/* kws-test-2/E/
cp logs/intermittent/KWS_CNN_S/* kws-test-2/I/
cp logs/energy/KWS_CNN_S/* kws-test-2/E/
