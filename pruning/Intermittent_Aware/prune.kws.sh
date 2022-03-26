Model='KWS'
PRUNE_METHOD=$1
COMMON_FLAGS='--arch '$Model' --batch-size 100 --test-batch-size 100 --lr 0.0005 --epochs 10000'
CANDIDATES_PRUNING_RATIOS='0.25 0.3 0.35 0.4'
MY_DEBUG='--debug -1' # -1: none, 0: info, 1: debug
PRUNE_COMMON_FLAGS='--prune '$PRUNE_METHOD' --sa '$MY_DEBUG
# original training -- 92.50%
python main.py $COMMON_FLAGS
'''
# 25% pruned -- 99.23/99.31
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 0 \
	--pretrained saved_models/HAR.origin.pth.tar
# 43.1% pruned -- 99.29/99.31
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 1 \
	--pretrained saved_models/$PRUNE_METHOD/$Model/stage_0.pth.tar
# 57.4% pruned -- 99.16/99.27
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 2 \
	--pretrained saved_models/$PRUNE_METHOD/$Model/stage_1.pth.tar
# 67.6% pruned -- 98.99/99.16
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 3 \
	--pretrained saved_models/$PRUNE_METHOD/$Model/stage_2.pth.tar
# 75.2% pruned -- 99.01/99.15
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 4 \
	--pretrained saved_models/$PRUNE_METHOD/$Model/stage_3.pth.tar
# 81.1% pruned -- 98.91(intermittent)/99.04
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 5 \
	--pretrained saved_models/$PRUNE_METHOD/$Model/stage_4.pth.tar
# xx% pruned -- 98.85/98.92(energy)
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 6 \
	--pretrained saved_models/$PRUNE_METHOD/$Model/stage_5.pth.tar
# xx% pruned -- 98./98.91(energy)
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 7 \
	--pretrained saved_models/$PRUNE_METHOD/$Model/stage_6.pth.tar
# xx% pruned -- 98.80/98.66
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 8 \
	--pretrained saved_models/$PRUNE_METHOD/$Model/stage_7.pth.tar
