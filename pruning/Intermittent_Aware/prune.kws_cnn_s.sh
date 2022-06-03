Model='KWS_CNN_S'
PRUNE_METHOD=$1
LEARNING_RATE_LIST='0.0005 0.0001 0.00002'
GPUS='0 1 2 3'
COMMON_FLAGS='--arch '$Model' --batch-size 16 --test-batch-size 16 --lr 0.0005 --epochs 150 --lr-epochs 50 --gpus '$GPUS' --learning_rate_list '$LEARNING_RATE_LIST
CANDIDATES_PRUNING_RATIOS='0 0 0 0 0'
MY_DEBUG='--debug 1' # -1: none, 0: info, 1: debug
PRUNE_COMMON_FLAGS='--prune '$PRUNE_METHOD' --sa '$MY_DEBUG' --overall-pruning-ratio 0.2'
SENSITIVITY_ANALYSIS_FLAGS='--arch '$Model' --batch-size 512 --test-batch-size 512 --lr 0.0005 --epochs 50 --lr-epochs 100 --gpus '$GPUS' --learning_rate_list '$LEARNING_RATE_LIST' --prune '$PRUNE_METHOD' --sen-ana'

# original training --- 87.83% (88.12%)
python main.py $COMMON_FLAGS
'''
# sensitivity analysis
python main.py $SENSITIVITY_ANALYSIS_FLAGS \
	--candidates-pruning-ratios $CANDIDATES_PRUNING_RATIOS \
	--stage 1 \
	--pretrained saved_models/$Model.origin.pth.tar
# 87.16%/86.83%
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 0 \
	--pretrained saved_models/$Model.origin.pth.tar
# 87.12%/86.87%
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 1 \
	--pretrained saved_models/$PRUNE_METHOD/$Model/stage_0.pth.tar
# 86.83%/86.69%
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 2 \
	--pretrained saved_models/$PRUNE_METHOD/$Model/stage_1.pth.tar
# 86.07% (intermittent)/85.66% (energy)
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 3 \
	--pretrained saved_models/$PRUNE_METHOD/$Model/stage_2.pth.tar
# 85.21/
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 4 \
	--pretrained saved_models/$PRUNE_METHOD/$Model/stage_3.pth.tar
#  / 81.39
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 5 \
	--pretrained saved_models/$PRUNE_METHOD/$Model/stage_4.pth.tar \
	--retrain
#  / 80.84
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 6 \
	--pretrained saved_models/$PRUNE_METHOD/$Model/stage_5.pth.tar
#  / 79.86
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 7 \
	--pretrained saved_models/$PRUNE_METHOD/$Model/stage_6.pth.tar
#  / 77.79
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 8 \
	--pretrained saved_models/$PRUNE_METHOD/$Model/stage_7.pth.tar
