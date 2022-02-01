COMMON_FLAGS='--arch LeNet_5'
PRUNE_METHOD='intermittent'
GROUP_SIZE='1 1 1 2'
CANDIDATES_PRUNING_RATIOS='0.25 0.3 0.35 0.4'

# original training -- 99.23%
# python main.py $COMMON_FLAGS

# 30.4% pruned -- 99.19%
python main.py $COMMON_FLAGS --stage 0 --prune $PRUNE_METHOD \
 	--group $GROUP_SIZE \
	--pretrained saved_models/LeNet_5.origin.pth.tar \
	--lr 0.1 --lr-epochs 15
# 51.6% pruned -- 99.17%
python main.py $COMMON_FLAGS --stage 1 --prune $PRUNE_METHOD \
	--group $GROUP_SIZE \
	--pretrained saved_models/$PRUNE_METHOD/LeNet_5.prune.group_size5.0.pth.tar \
	--lr 0.01 --lr-epochs 20

# 66.5% pruned -- 98.97%
python main.py $COMMON_FLAGS --stage 2 --prune $PRUNE_METHOD \
	--group $GROUP_SIZE \
	--pretrained saved_models/$PRUNE_METHOD/LeNet_5.prune.group_size5.1.pth.tar \
	--lr 0.01 --lr-epochs 20

# 76.8% pruned -- 98.94%
python main.py $COMMON_FLAGS --stage 3 --prune $PRUNE_METHOD \
	--group $GROUP_SIZE \
	--pretrained saved_models/$PRUNE_METHOD/LeNet_5.prune.group_size5.2.pth.tar \
	--lr 0.01 --lr-epochs 20

# 83.9% pruned -- 98.92%
python main.py $COMMON_FLAGS --stage 4 --prune $PRUNE_METHOD \
	--group $GROUP_SIZE \
	--pretrained saved_models/$PRUNE_METHOD/LeNet_5.prune.group_size5.3.pth.tar \
	--lr 0.01 --lr-epochs 20

# 98.61% pruned -- 98.61%
python main.py $COMMON_FLAGS --stage 5 --prune $PRUNE_METHOD \
	--group $GROUP_SIZE \
	--pretrained saved_models/$PRUNE_METHOD/LeNet_5.prune.group_size5.4.pth.tar \
	--lr 0.01 --lr-epochs 20
