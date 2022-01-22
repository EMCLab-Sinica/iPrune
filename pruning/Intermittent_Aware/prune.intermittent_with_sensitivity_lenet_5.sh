COMMON_FLAGS='--arch LeNet_5'
GROUP_SIZE='1 1 1 5'

# original training -- 99.23%
# python main.py $COMMON_FLAGS

# 43.1% pruned -- 99.25%
python main.py $COMMON_FLAGS --with_sen --prune 'intermittent' --stage 0 --group $GROUP_SIZE \
	--pretrained saved_models/LeNet_5.origin.pth.tar \
	--lr 0.1 --lr-epochs 15 --pruning_ratio 0.3

# 65.5% pruned -- 99.13%
python main.py $COMMON_FLAGS --with_sen --prune 'intermittent' --stage 1 --group $GROUP_SIZE\
	--pretrained saved_models/with_sensitivity/LeNet_5.prune.intermittent.group_size5.0.pth.tar \
	--lr 0.01 --lr-epochs 20 --pruning_ratio 0.4

# 77.9% pruned -- 99.11%
python main.py $COMMON_FLAGS --with_sen --prune 'intermittent' --stage 2 --group $GROUP_SIZE\
	--pretrained saved_models/with_sensitivity/LeNet_5.prune.intermittent.group_size5.1.pth.tar \
	--lr 0.01 --lr-epochs 20 --pruning_ratio 0.5

# 85.2% pruned -- 99.09%
python main.py $COMMON_FLAGS --with_sen --prune 'intermittent' --stage 3 --group $GROUP_SIZE\
	--pretrained saved_models/with_sensitivity/LeNet_5.prune.intermittent.group_size5.2.pth.tar \
	--lr 0.01 --lr-epochs 20 --pruning_ratio 0.6

# 89.5% pruned -- 98.97%
#python main.py $COMMON_FLAGS --with_sen --prune 'intermittent' --stage 4 --group $GROUP_SIZE\
#	--pretrained saved_models/with_sensitivity/LeNet_5.prune.intermittent.group_size5.3.pth.tar \
#	--lr 0.01 --lr-epochs 20 --pruning_ratio 0.7

# 92.6% pruned -- 98.61%
#python main.py $COMMON_FLAGS --with_sen --prune 'intermittent' --stage 5 --group $GROUP_SIZE\
#	--pretrained saved_models/with_sensitivity/LeNet_5.prune.intermittent.group_size5.4.pth.tar \
#	--lr 0.01 --lr-epochs 20 --pruning_ratio 0.8

# 94.8% pruned -- 97.23%
#python main.py $COMMON_FLAGS --with_sen --prune 'intermittent' --stage 6 --group $GROUP_SIZE\
#	--pretrained saved_models/with_sensitivity/LeNet_5.prune.intermittent.group_size5.5.pth.tar \
#	--lr 0.01 --lr-epochs 20 --pruning_ratio 0.9
