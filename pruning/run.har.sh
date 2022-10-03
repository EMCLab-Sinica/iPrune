#!/bin/sh
echo "Start HAR model"
bash prune.har.I_2.sh > new_data/[HAR]Prune_intermittent_2_log.txt
echo "[HAR] Prune_intermittent_2"
bash prune.har.E_2.sh > new_data/[HAR]Prune_energy_2_log.txt
echo "[HAR] Prune_energy_2"

bash prune.har.I_3.sh > new_data/[HAR]Prune_intermittent_3_log.txt
echo "[HAR] Prune_intermittent_3"
bash prune.har.E_3.sh > new_data/[HAR]Prune_energy_3_log.txt
echo "[HAR] Prune_energy_3"

echo "Start KWS_CNN mode-STAGE 2"
bash prune.kws_cnn_s.I_2.sh > new_data/[KWS_CNN]Prune_intermittent_2_log.txt
echo "[KWS_CNN] Prune_intermittent_2"
bash prune.kws_cnn_s.E_2.sh > new_data/[KWS_CNN]Prune_energy_2_log.txt
echo "[KWS_CNN] Prune_energy_2"
