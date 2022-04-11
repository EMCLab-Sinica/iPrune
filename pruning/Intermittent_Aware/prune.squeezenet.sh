Model='SqueezeNet'
PRUNE_METHOD=$1
LEARNING_RATE_LIST='0.001 0.0005'
COMMON_FLAGS='--arch SqueezeNet --batch-size 32 --test-batch-size 32 --lr 0.001 --lr-epochs 75 --learning_rate_list '$LEARNING_RATE_LIST
CANDIDATES_PRUNING_RATIOS='0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5'
MY_DEBUG='--debug 1'
PRUNE_COMMON_FLAGS='--prune '$PRUNE_METHOD' --sa '$MY_DEBUG

'''
# original training --75.76%
python main.py $COMMON_FLAGS --epochs 300
# 76.65%/76.48%
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 0 \
	--pretrained saved_models/$Model.origin.pth.tar \
	--epochs 150 \
	--candidates-pruning-ratios $CANDIDATES_PRUNING_RATIOS
# 76.75/77.13%
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 1 \
	--pretrained saved_models/$PRUNE_METHOD/$Model/stage_0.pth.tar \
	--epochs 150 \
	--candidates-pruning-ratios $CANDIDATES_PRUNING_RATIOS
# 76.33/76.48
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 2 \
	--pretrained saved_models/$PRUNE_METHOD/$Model/stage_1.pth.tar \
	--epochs 150 \
	--candidates-pruning-ratios $CANDIDATES_PRUNING_RATIOS
'''
# 75.59 (intermittent)/75.35(energy)
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 3 \
	--pretrained saved_models/$PRUNE_METHOD/$Model/stage_2.pth.tar \
	--epochs 150 \
	--candidates-pruning-ratios $CANDIDATES_PRUNING_RATIOS
'''
# xxx/xxx
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 4 \
	--pretrained saved_models/$PRUNE_METHOD/$Model/stage_3.pth.tar \
	--epochs 150 \
	--candidates-pruning-ratios $CANDIDATES_PRUNING_RATIOS
# xxx/xxx
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 5 \
	--pretrained saved_models/$PRUNE_METHOD/$Model/stage_4.pth.tar \
	--epochs 150 \
	--candidates-pruning-ratios $CANDIDATES_PRUNING_RATIOS
