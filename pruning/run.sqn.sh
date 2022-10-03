#!/bin/sh
echo "Start SQN model"

bash prune.squeezenet.sh > new_data/[SQN]Pretrained_log.txt
echo "[SQN] Pretrained model"
bash prune.squeezenet.I_0.sh > new_data/[SQN]Prune_intermittent_0_log.txt
echo "[SQN] Prune_intermittent_0"
bash prune.squeezenet.I_1.sh > new_data/[SQN]Prune_intermittent_1_log.txt
echo "[SQN] Prune_intermittent_1"
bash prune.squeezenet.E_0.sh > new_data/[SQN]Prune_energy_0_log.txt
echo "[SQN] Prune_energy_0"
bash prune.squeezenet.E_1.sh > new_data/[SQN]Prune_energy_1_log.txt
echo "[SQN] Prune_energy_1"
