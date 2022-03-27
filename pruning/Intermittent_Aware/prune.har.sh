Model='HAR'
PRUNE_METHOD=$1
COMMON_FLAGS='--arch '$Model' --batch-size 600 --test-batch-size 600 --lr 0.0001 --epochs 1000'
CANDIDATES_PRUNING_RATIOS='0.25 0.3 0.35 0.4'
MY_DEBUG='--debug 1' # -1: none, 0: info, 1: debug
PRUNE_COMMON_FLAGS='--prune '$PRUNE_METHOD' --sa '$MY_DEBUG
			    # intermittent/energy
'''
# original training -- 92.50%
python main.py $COMMON_FLAGS
# 25% pruned -- 92.16/92.20
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 0 \
	--pretrained saved_models/HAR.origin.pth.tar
# 43.1% pruned -- 92.37/92.67
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 1 \
	--pretrained saved_models/$PRUNE_METHOD/$Model/stage_0.pth.tar
# 57.4% pruned -- 92.26 (intermittent)/92.03 (energy)
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 2 \
	--pretrained saved_models/$PRUNE_METHOD/$Model/stage_1.pth.tar
# 67.6% pruned -- 92.70/91.38
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 3 \
	--pretrained saved_models/$PRUNE_METHOD/$Model/stage_2.pth.tar
'''
# 75.2% pruned -- xxxx/91.82
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 4 \
	--pretrained saved_models/$PRUNE_METHOD/$Model/stage_3.pth.tar
# 81.1% pruned -- xxxx(intermittent)/91.86
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 5 \
	--pretrained saved_models/$PRUNE_METHOD/$Model/stage_4.pth.tar
