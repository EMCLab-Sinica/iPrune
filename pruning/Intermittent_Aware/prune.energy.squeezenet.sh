Model='SqueezeNet'
COMMON_FLAGS='--arch SqueezeNet'
PRUNE_METHOD='energy'
GROUP_SIZE='1 1 1 2'
CANDIDATES_PRUNING_RATIOS='0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5'

# original training --73.93%
# python main.py $COMMON_FLAGS --batch-size 32 --test-batch-size 32 --epochs 300 --lr 0.001

# 35.3% pruned -- 72.99%
python main.py $COMMON_FLAGS --stage 0 --prune $PRUNE_METHOD --group $GROUP_SIZE \
	--pretrained saved_models/$Model.origin.pth.tar \
	--lr 0.001 \
	--batch-size 32 \
	--test-batch-size 32 \
	--epochs 50 \
	--candidates-pruning-ratios $CANDIDATES_PRUNING_RATIOS

# 57.4% pruned -- 73.74%
python main.py $COMMON_FLAGS --stage 1 --prune $PRUNE_METHOD --group $GROUP_SIZE\
	--pretrained saved_models/$PRUNE_METHOD/$Model.prune.group_size5.0.pth.tar \
	--lr 0.001 \
	--batch-size 32 \
	--test-batch-size 32 \
	--epochs 50 \
	--candidates-pruning-ratios $CANDIDATES_PRUNING_RATIOS

# 70.3% pruned -- 70.30%
python main.py $COMMON_FLAGS --prune $PRUNE_METHOD --stage 2 --group $GROUP_SIZE\
	--pretrained saved_models/$PRUNE_METHOD/$Model.prune.group_size5.1.pth.tar \
	--lr 0.001 \
	--batch-size 32 \
	--test-batch-size 32 \
	--epochs 50 \
	--candidates-pruning-ratios $CANDIDATES_PRUNING_RATIOS


# 78.8% pruned -- 74.06%
python main.py $COMMON_FLAGS --prune $PRUNE_METHOD --stage 3 --group $GROUP_SIZE\
	--pretrained saved_models/$PRUNE_METHOD/$Model.prune.group_size5.2.pth.tar \
	--lr 0.001 \
	--batch-size 32 \
	--test-batch-size 32 \
	--epochs 50 \
	--candidates-pruning-ratios $CANDIDATES_PRUNING_RATIOS


# 83.9% pruned -- 72.59%
python main.py $COMMON_FLAGS --prune $PRUNE_METHOD --stage 4 --group $GROUP_SIZE\
	--pretrained saved_models/$PRUNE_METHOD/$Model.prune.group_size5.3.pth.tar \
	--lr 0.001 \
	--batch-size 32 \
	--test-batch-size 32 \
	--epochs 50 \
	--candidates-pruning-ratios $CANDIDATES_PRUNING_RATIOS


# 87.7% pruned -- 72.18%
python main.py $COMMON_FLAGS --prune $PRUNE_METHOD --stage 5 --group $GROUP_SIZE\
	--pretrained saved_models/$PRUNE_METHOD/$Model.prune.group_size5.4.pth.tar \
	--lr 0.001 \
	--batch-size 32 \
	--test-batch-size 32 \
	--epochs 50 \
	--candidates-pruning-ratios $CANDIDATES_PRUNING_RATIOS

