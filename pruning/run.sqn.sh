#!/bin/sh

echo "[SQN]"

bash prune.squeezenet.sh > new_data/[SQN]Prune_intermittent_pretrained_log.txt
echo "[SQN] Pretrained"



bash prune.squeezenet.sh > new_data/[SQN]Prune_intermittent_pretrained_log.txt
echo "[SQN] Pretrained"


echo "[SQN]-R=0.1"

bash prune.squeezenet_R1.I_0.sh > new_data/[SQN]-R1-Prune_intermittent_0_log.txt
echo "[SQN]-R1- Prune_intermittent_0"
bash prune.squeezenet_R1.E_0.sh > new_data/[SQN]-R1-Prune_energy_0_log.txt
echo "[SQN]-R1- Prune_energy_0"

bash prune.squeezenet_R1.I_1.sh > new_data/[SQN]-R1-Prune_intermittent_1_log.txt
echo "[SQN]-R1- Prune_intermittent_1"
bash prune.squeezenet_R1.E_1.sh > new_data/[SQN]-R1-Prune_energy_1_log.txt
echo "[SQN]-R1- Prune_energy_1"

bash prune.squeezenet_R1.I_2.sh > new_data/[SQN]-R1-Prune_intermittent_2_log.txt
echo "[SQN]-R1- Prune_intermittent_2"
bash prune.squeezenet_R1.E_2.sh > new_data/[SQN]-R1-Prune_energy_2_log.txt
echo "[SQN]-R1- Prune_energy_2"

bash prune.squeezenet_R1.I_3.sh > new_data/[SQN]-R1-Prune_intermittent_3_log.txt
echo "[SQN]-R1- Prune_intermittent_3"
bash prune.squeezenet_R1.E_3.sh > new_data/[SQN]-R1-Prune_energy_3_log.txt
echo "[SQN]-R1- Prune_energy_3"

bash prune.squeezenet_R1.I_4.sh > new_data/[SQN]-R1-Prune_intermittent_4_log.txt
echo "[SQN]-R1- Prune_intermittent_4"
bash prune.squeezenet_R1.E_4.sh > new_data/[SQN]-R1-Prune_energy_4_log.txt
echo "[SQN]-R1- Prune_energy_4"

bash prune.squeezenet_R1.I_5.sh > new_data/[SQN]-R1-Prune_intermittent_5_log.txt
echo "[SQN]-R1- Prune_intermittent_5"
bash prune.squeezenet_R1.E_5.sh > new_data/[SQN]-R1-Prune_energy_5_log.txt
echo "[SQN]-R1- Prune_energy_5"

bash prune.squeezenet_R1.I_6.sh > new_data/[SQN]-R1-Prune_intermittent_6_log.txt
echo "[SQN]-R1- Prune_intermittent_6"
bash prune.squeezenet_R1.E_6.sh > new_data/[SQN]-R1-Prune_energy_6_log.txt
echo "[SQN]-R1- Prune_energy_6"

mkdir SqueezeNet-test-R1
mkdir SqueezeNet-test-R1/I
mkdir SqueezeNet-test-R1/E


cp logs/intermittent/SqueezeNet/* SqueezeNet-test-R1/I/
cp logs/energy/SqueezeNet/* SqueezeNet-test-R1/E/
cp saved_models/SqueezeNet.origin.pth.tar SqueezeNet-test-R1/
cp saved_models/intermittent/SqueezeNet/* SqueezeNet-test-R1/I/
cp saved_models/energy/SqueezeNet/* SqueezeNet-test-R1/E/


echo "[SQN]-R=0.3"

bash prune.squeezenet_R3.I_0.sh > new_data/[SQN]-R3-Prune_intermittent_0_log.txt
echo "[SQN]-R3- Prune_intermittent_0"
bash prune.squeezenet_R3.E_0.sh > new_data/[SQN]-R3-Prune_energy_0_log.txt
echo "[SQN]-R3- Prune_energy_0"

bash prune.squeezenet_R3.I_1.sh > new_data/[SQN]-R3-Prune_intermittent_1_log.txt
echo "[SQN]-R3- Prune_intermittent_1"
bash prune.squeezenet_R3.E_1.sh > new_data/[SQN]-R3-Prune_energy_1_log.txt
echo "[SQN]-R3- Prune_energy_1"

bash prune.squeezenet_R3.I_2.sh > new_data/[SQN]-R3-Prune_intermittent_2_log.txt
echo "[SQN]-R3- Prune_intermittent_2"
bash prune.squeezenet_R3.E_2.sh > new_data/[SQN]-R3-Prune_energy_2_log.txt
echo "[SQN]-R3- Prune_energy_2"

bash prune.squeezenet_R3.I_3.sh > new_data/[SQN]-R3-Prune_intermittent_3_log.txt
echo "[SQN]-R3- Prune_intermittent_3"
bash prune.squeezenet_R3.E_3.sh > new_data/[SQN]-R3-Prune_energy_3_log.txt
echo "[SQN]-R3- Prune_energy_3"

mkdir SqueezeNet-test-R3
mkdir SqueezeNet-test-R3/I
mkdir SqueezeNet-test-R3/E


cp logs/intermittent/SqueezeNet/* SqueezeNet-test-R3/I/
cp logs/energy/SqueezeNet/* SqueezeNet-test-R3/E/
cp saved_models/SqueezeNet.origin.pth.tar SqueezeNet-test-R3/
cp saved_models/intermittent/SqueezeNet/* SqueezeNet-test-R3/I/
cp saved_models/energy/SqueezeNet/* SqueezeNet-test-R3/E/


echo "[SQN]-R=0.2"

bash prune.squeezenet_R2.I_0.sh > new_data/[SQN]-R2-Prune_intermittent_0_log.txt
echo "[SQN]-R2- Prune_intermittent_0"
bash prune.squeezenet_R2.E_0.sh > new_data/[SQN]-R2-Prune_energy_0_log.txt
echo "[SQN]-R2- Prune_energy_0"

bash prune.squeezenet_R2.I_1.sh > new_data/[SQN]-R2-Prune_intermittent_1_log.txt
echo "[SQN]-R2- Prune_intermittent_1"
bash prune.squeezenet_R2.E_1.sh > new_data/[SQN]-R2-Prune_energy_1_log.txt
echo "[SQN]-R2- Prune_energy_1"

bash prune.squeezenet_R2.I_2.sh > new_data/[SQN]-R2-Prune_intermittent_2_log.txt
echo "[SQN]-R2- Prune_intermittent_2"
bash prune.squeezenet_R2.E_2.sh > new_data/[SQN]-R2-Prune_energy_2_log.txt
echo "[SQN]-R2- Prune_energy_2"

bash prune.squeezenet_R2.I_3.sh > new_data/[SQN]-R2-Prune_intermittent_3_log.txt
echo "[SQN]-R2- Prune_intermittent_3"
bash prune.squeezenet_R2.E_3.sh > new_data/[SQN]-R2-Prune_energy_3_log.txt
echo "[SQN]-R2- Prune_energy_3"

mkdir SqueezeNet-test-R2
mkdir SqueezeNet-test-R2/I
mkdir SqueezeNet-test-R2/E


cp logs/intermittent/SqueezeNet/* SqueezeNet-test-R2/I/
cp logs/energy/SqueezeNet/* SqueezeNet-test-R2/E/
cp saved_models/SqueezeNet.origin.pth.tar SqueezeNet-test-R2/
cp saved_models/intermittent/SqueezeNet/* SqueezeNet-test-R2/I/
cp saved_models/energy/SqueezeNet/* SqueezeNet-test-R2/E/
