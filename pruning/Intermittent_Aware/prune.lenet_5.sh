Model='LeNet_5'
PRUNE_METHOD=$1
GROUP_SIZE='1 1 1 2'
COMMON_FLAGS='--arch LeNet_5'
CANDIDATES_PRUNING_RATIOS='0.25 0.3 0.35 0.4'
MY_DEBUG='--debug 1' # -1: none, 0: info, 1: debug
PRUNE_COMMON_FLAGS='--prune '$PRUNE_METHOD' --group '$GROUP_SIZE' --sa '$MY_DEBUG' --lr 0.01 --lr-epochs 20'

# original training -- 99.23%
# python main.py $COMMON_FLAGS
# 35% pruned -- 99.19%
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 0 \
	--pretrained saved_models/LeNet_5.origin.pth.tar
'''
# 57.6% pruned -- 99.17%
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 1 \
	--pretrained saved_models/$PRUNE_METHOD/$Model/stage_0.pth.tar
# xx% pruned -- 99.06%
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 2 \
	--pretrained saved_models/$PRUNE_METHOD/$Model/stage_1.pth.tar

# xx% pruned -- 98.99% (end)
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 3 \
	--pretrained saved_models/$PRUNE_METHOD/$Model/stage_2.pth.tar
# 83.9% pruned -- 98.92%
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 4 \
	--pretrained saved_models/$PRUNE_METHOD/$Model/stage_3.pth.tar
# 98.61% pruned -- 98.61%
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 5 \
	--pretrained saved_models/$PRUNE_METHOD/$Model/stage_4.pth.tar
