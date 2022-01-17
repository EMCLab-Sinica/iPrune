COMMON_FLAGS='--arch LeNet_5'
GROUP_SIZE='1 1 1 5'

# original training -- 99.18%
# python main.py $COMMON_FLAGS

# 51.5% pruned -- 99.26%
#python main.py $COMMON_FLAGS --prune 'intermittent' --stage 0 \
# 	--group $GROUP_SIZE \
#	--pretrained saved_models/LeNet_5.origin.pth.tar \
#	--lr 0.1 --lr-epochs 15 --pruning_ratio 0.3

# 76.9% pruned -- 98.95%
#python main.py $COMMON_FLAGS --stage 1 --prune 'intermittent' --group $GROUP_SIZE\
#	--pretrained saved_models/with_sensitivity/LeNet_5.prune.intermittent.group_size5.0.pth.tar \
#	--lr 0.01 --lr-epochs 20 --pruning_ratio 0.4

# 88.2% pruned -- 98.98%
#python main.py $COMMON_FLAGS --prune 'intermittent' --stage 2 --group $GROUP_SIZE\
#	--pretrained saved_models/with_sensitivity/LeNet_5.prune.intermittent.group_size5.1.pth.tar \
#	--lr 0.01 --lr-epochs 20 --pruning_ratio 0.5

# 93.3% pruned -- 98.70%
#python main.py $COMMON_FLAGS --prune 'intermittent' --stage 3 --group $GROUP_SIZE\
#	--pretrained saved_models/with_sensitivity/LeNet_5.prune.intermittent.group_size5.2.pth.tar \
#	--lr 0.01 --lr-epochs 20 --pruning_ratio 0.6

# 95.4% pruned -- 86.37%
# python main.py $COMMON_FLAGS --prune 'intermittent' --stage 4 --group $GROUP_SIZE\
#	--pretrained saved_models/with_sensitivity/LeNet_5.prune.intermittent.group_size5.3.pth.tar \
#	--lr 0.01 --lr-epochs 20 --pruning_ratio 0.7



# Reduce each steps
# 60.5% pruned -- 99.13%
#python main.py $COMMON_FLAGS --prune 'intermittent' --stage 1 --group $GROUP_SIZE\
#	--pretrained saved_models/with_sensitivity/LeNet_5.prune.intermittent.group_size5.0.pth.tar \
#	--lr 0.01 --lr-epochs 15 --pruning_ratio 0.3

# 74.0% pruned -- 99.08%
#python main.py $COMMON_FLAGS --prune 'intermittent' --stage 2 --group $GROUP_SIZE\
#	--pretrained saved_models/with_sensitivity/LeNet_5.prune.intermittent.group_size5.1.pth.tar \
#	--lr 0.01 --lr-epochs 20 --pruning_ratio 0.4

# 81.5% pruned -- 98.98%
#python main.py $COMMON_FLAGS --prune 'intermittent' --stage 3 --group $GROUP_SIZE\
#	--pretrained saved_models/with_sensitivity/LeNet_5.prune.intermittent.group_size5.2.pth.tar \
#	--lr 0.01 --lr-epochs 20 --pruning_ratio 0.5

# 86.8% pruned -- 99.13%
#python main.py $COMMON_FLAGS --prune 'intermittent' --stage 4 --group $GROUP_SIZE\
#	--pretrained saved_models/with_sensitivity/LeNet_5.prune.intermittent.group_size5.3.pth.tar \
#	--lr 0.01 --lr-epochs 20 --pruning_ratio 0.6

# 88.1% pruned -- 99.11#%
#python main.py $COMMON_FLAGS --prune 'intermittent' --stage 5 --group $GROUP_SIZE\
#	--pretrained saved_models/with_sensitivity/LeNet_5.prune.intermittent.group_size5.4.pth.tar \
#	--lr 0.01 --lr-epochs 20 --pruning_ratio 0.7

# 91.9% pruned -- 98.72%
python main.py $COMMON_FLAGS --prune 'intermittent' --stage 6 --group $GROUP_SIZE\
	--pretrained saved_models/with_sensitivity/LeNet_5.prune.intermittent.group_size5.4.pth.tar \
	--lr 0.01 --lr-epochs 20 --pruning_ratio 0.8

# 94.2% pruned -- 97.82%
python main.py $COMMON_FLAGS --prune 'intermittent' --stage 7 --group $GROUP_SIZE\
	--pretrained saved_models/with_sensitivity/LeNet_5.prune.intermittent.group_size5.6.pth.tar \
	--lr 0.01 --lr-epochs 20 --pruning_ratio 0.9
