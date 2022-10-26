#!/bin/sh

bash prune.kws_cnn_s.sh -e -v 0 -o 0.1 -a 0 > log_data/[KWS_CNN]-R1-Prune_intermittent_0_log_debug_r.txt
bash prune.kws_cnn_s.sh -e -v 1 -o 0.2 -a 0 > log_data/[KWS_CNN]-R2-Prune_intermittent_0_log_debug_r.txt
bash prune.kws_cnn_s.sh -e -v 2 -o 0.3 -a 0 > log_data/[KWS_CNN]-R3-Prune_intermittent_0_log_debug_r.txt
bash prune.kws_cnn_s.sh -e -v 3 -o 0.4 -a 0 > log_data/[KWS_CNN]-R4-Prune_intermittent_0_log_debug_r.txt
