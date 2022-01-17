COMMON_FLAGS='--arch LeNet_5'
GROUP_SIZE='1 1 1 5'

# original training -- 99.18%
# python main.py $COMMON_FLAGS

# 20.7% pruned -- 99.22%
# python main.py $COMMON_FLAGS --prune 'intermittent' --stage 0 \
# 	--group $GROUP_SIZE \
#	--pretrained saved_models/LeNet_5.origin.pth.tar \
#	--lr 0.1 --lr-epochs 15

# 37.1% pruned -- 99.24%
# python main.py $COMMON_FLAGS --stage 1 --prune 'intermittent' --group $GROUP_SIZE\
#	--pretrained saved_models/LeNet_5.prune.intermittent.group_size5.0.pth.tar \
#	--lr 0.01 --lr-epochs 20

# 50.0% pruned -- 99.07%
# python main.py $COMMON_FLAGS --prune 'intermittent' --stage 2 --group $GROUP_SIZE\
#	--pretrained saved_models/LeNet_5.prune.intermittent.group_size5.1.pth.tar \
#	--lr 0.01 --lr-epochs 20

# 60.3% pruned -- 98.85%
# python main.py $COMMON_FLAGS --prune 'intermittent' --stage 3 --group $GROUP_SIZE\
#	--pretrained saved_models/LeNet_5.prune.intermittent.group_size5.2.pth.tar \
#	--lr 0.01 --lr-epochs 20

# 68.4% pruned -- 98.76%
# python main.py $COMMON_FLAGS --prune 'intermittent' --stage 4 --group $GROUP_SIZE\
#	--pretrained saved_models/LeNet_5.prune.intermittent.group_size5.3.pth.tar \
#	--lr 0.01 --lr-epochs 20
