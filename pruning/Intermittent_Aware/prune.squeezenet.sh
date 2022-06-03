Model='SqueezeNet'
PRUNE_METHOD=$1
LEARNING_RATE_LIST='0.001 0.0005'
GPUS='0 1 2 3'
COMMON_FLAGS='--arch '$Model' --batch-size 600 --test-batch-size 600 --lr 0.0001 --epochs 300 --lr-epochs 150 --gpus '$GPUS' --learning_rate_list '$LEARNING_RATE_LIST
CANDIDATES_PRUNING_RATIOS='0 0 0 0 0 0 0 0 0 0 0'
MY_DEBUG='--debug 1'
PRUNE_COMMON_FLAGS='--prune '$PRUNE_METHOD' --sa '$MY_DEBUG' --overall-pruning-ratio 0.2'
SENSITIVITY_ANALYSIS_FLAGS='--arch '$Model' --batch-size 128 --test-batch-size 128 --lr 0.0005 --epochs 100 --lr-epochs 50 --gpus '$GPUS' --learning_rate_list '$LEARNING_RATE_LIST' --prune '$PRUNE_METHOD' --sen-ana'

# original training --75.76%
python main.py $COMMON_FLAGS
'''
# sensitivity analysis
python main.py $SENSITIVITY_ANALYSIS_FLAGS \
	--candidates-pruning-ratios $CANDIDATES_PRUNING_RATIOS \
	--stage 0 \
	--pretrained saved_models/$Model.origin.pth.tar
# 76.65%/76.48%
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 0 \
	--pretrained saved_models/$Model.origin.pth.tar \
	--candidates-pruning-ratios $CANDIDATES_PRUNING_RATIOS
# 76.75/77.13%
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 1 \
	--pretrained saved_models/$PRUNE_METHOD/$Model/stage_0.pth.tar \
	--candidates-pruning-ratios $CANDIDATES_PRUNING_RATIOS
# 76.33/76.48
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 2 \
	--pretrained saved_models/$PRUNE_METHOD/$Model/stage_1.pth.tar \
	--candidates-pruning-ratios $CANDIDATES_PRUNING_RATIOS
# 75.59 (intermittent)/75.35(energy)
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 3 \
	--pretrained saved_models/$PRUNE_METHOD/$Model/stage_2.pth.tar \
	--candidates-pruning-ratios $CANDIDATES_PRUNING_RATIOS
# xxx/xxx
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 4 \
	--pretrained saved_models/$PRUNE_METHOD/$Model/stage_3.pth.tar \
	--candidates-pruning-ratios $CANDIDATES_PRUNING_RATIOS
# xxx/xxx
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 5 \
	--pretrained saved_models/$PRUNE_METHOD/$Model/stage_4.pth.tar \
	--candidates-pruning-ratios $CANDIDATES_PRUNING_RATIOS
