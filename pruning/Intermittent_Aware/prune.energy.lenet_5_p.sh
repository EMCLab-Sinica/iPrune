COMMON_FLAGS='--arch LeNet_5_p'
GROUP_SIZE='1 1 1 2'
CANDIDATES_PRUNING_RATIOS='0.25 0.3 0.35 0.4'
PRUNE_METHOD='energy'
# original training -- 99.20%
#python main.py $COMMON_FLAGS --lr 1 --batch-size 64 --test-batch-size 1000 --epochs 30

# 30.4% pruned -- 99.19%
python main.py $COMMON_FLAGS --prune $PRUNE_METHOD --stage 0 \
	--group $GROUP_SIZE \
	--pretrained saved_models/LeNet_5_p.origin1.pth.tar \
	--lr 1 --epochs 20 \
	--batch-size 64 --test-batch-size 1000

# 51.6% pruned -- 99.17%
python main.py $COMMON_FLAGS --prune $PRUNE_METHOD --stage 1 \
	--group $GROUP_SIZE \
	--pretrained saved_models/$PRUNE_METHOD/LeNet_5_p.prune.group_size5.0.pth.tar \
	--lr 1 --epochs 20 \
	--batch-size 64 --test-batch-size 1000

# 66.5% pruned -- 98.97%
python main.py $COMMON_FLAGS --prune $PRUNE_METHOD --stage 2 \
	--group $GROUP_SIZE \
	--pretrained saved_models/$PRUNE_METHOD/LeNet_5_p.prune.group_size5.1.pth.tar \
	--lr 1 --epochs 20 \
	--batch-size 64 --test-batch-size 1000

# 76.8% pruned -- 98.94%
python main.py $COMMON_FLAGS --prune $PRUNE_METHOD --stage 3 \
	--group $GROUP_SIZE \
	--pretrained saved_models/$PRUNE_METHOD/LeNet_5_p.prune.group_size5.2.pth.tar \
	--lr 1 --epochs 20 \
	--batch-size 64 --test-batch-size 1000

# 83.9% pruned -- 98.92%
python main.py $COMMON_FLAGS --prune $PRUNE_METHOD --stage 4 \
	--group $GROUP_SIZE \
	--pretrained saved_models/$PRUNE_METHOD/LeNet_5_p.prune.group_size5.3.pth.tar \
	--lr 1 --epochs 20 \
	--batch-size 64 --test-batch-size 1000

# 98.61% pruned -- 98.61%
python main.py $COMMON_FLAGS --prune $PRUNE_METHOD --stage 5 \
	--group $GROUP_SIZE \
	--pretrained saved_models/$PRUNE_METHOD/LeNet_5_p.prune.group_size5.4.pth.tar \
	--lr 1 --epochs 20 \
	--batch-size 64 --test-batch-size 1000
