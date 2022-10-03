#!/bin/sh
echo "Start KWS_CNN model"

bash prune.kws_cnn_s.sh > new_data/[KWS_CNN]Pretrained_log.txt
echo "[KWS_CNN] Pretrained model"
bash prune.kws_cnn_s.I_0.sh > new_data/[KWS_CNN]Prune_intermittent_0_log.txt
echo "[KWS_CNN] Prune_intermittent_0"
bash prune.kws_cnn_s.I_1.sh > new_data/[KWS_CNN]Prune_intermittent_1_log.txt
echo "[KWS_CNN] Prune_intermittent_1"
bash prune.kws_cnn_s.E_0.sh > new_data/[KWS_CNN]Prune_energy_0_log.txt
echo "[KWS_CNN] Prune_energy_0"
bash prune.kws_cnn_s.E_1.sh > new_data/[KWS_CNN]Prune_energy_1_log.txt
echo "[KWS_CNN] Prune_energy_1"
