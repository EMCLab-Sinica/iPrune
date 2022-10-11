#!/bin/sh


bash prune.squeezenet.I_2.sh > new_data/[SQN]Prune_intermittent_2_log.txt
echo "[SQN] Prune_intermittent_2"
bash prune.squeezenet.E_2.sh > new_data/[SQN]Prune_energy_2_log.txt
echo "[SQN] Prune_energy_2"


echo "Start SQN model-2"
bash prune.squeezenet.sh > new_data/[SQN]Pretrained_log-2.txt
echo "[SQN] Pretrained model-2"
