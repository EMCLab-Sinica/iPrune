Model='SqueezeNet'
PRUNE_METHOD=$1
GROUP_SIZE='1 1 1 2'
COMMON_FLAGS='--arch SqueezeNet --batch-size 32 --test-batch-size 32 --lr 0.001'
CANDIDATES_PRUNING_RATIOS='0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5'
MY_DEBUG='--debug 1'
PRUNE_COMMON_FLAGS='--prune '$PRUNE_METHOD' --group '$GROUP_SIZE' --sa '$MY_DEBUG

# original training --73.93%
# python main.py $COMMON_FLAGS --epochs 300
# 35.3% pruned -- 72.99%
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 0 \
	--pretrained saved_models/$Model.origin.pth.tar \
	--epochs 50 \
	--candidates-pruning-ratios $CANDIDATES_PRUNING_RATIOS
'''
# 57.4% pruned -- 73.74%
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 1 \
	--pretrained saved_models/$PRUNE_METHOD/$Model/stage_0.pth.tar \
	--epochs 50 \
	--candidates-pruning-ratios $CANDIDATES_PRUNING_RATIOS

# 70.3% pruned -- 70.30%
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 2 \
	--pretrained saved_models/$PRUNE_METHOD/$Model/stage_1.pth.tar \
	--epochs 50 \
	--candidates-pruning-ratios $CANDIDATES_PRUNING_RATIOS

# 78.8% pruned -- 74.06%
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 3 \
	--pretrained saved_models/$PRUNE_METHOD/$Model/stage_2.pth.tar \
	--epochs 50 \
	--candidates-pruning-ratios $CANDIDATES_PRUNING_RATIOS

# 83.9% pruned -- 72.59%
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 4 \
	--pretrained saved_models/$PRUNE_METHOD/$Model/stage_3.pth.tar \
	--epochs 60 \
	--candidates-pruning-ratios $CANDIDATES_PRUNING_RATIOS

# 87.7% pruned -- 72.18%
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 5 \
	--pretrained saved_models/$PRUNE_METHOD/$Model/stage_4.pth.tar \
	--epochs 50 \
	--candidates-pruning-ratios $CANDIDATES_PRUNING_RATIOS
