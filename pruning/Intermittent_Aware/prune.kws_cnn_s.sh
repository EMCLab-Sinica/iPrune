Model='KWS_CNN_S'
PRUNE_METHOD=$1
LEARNING_RATE_LIST='0.0005 0.0001 0.00002'
COMMON_FLAGS='--arch '$Model' --batch-size 32 --test-batch-size 32 --lr 0.0005 --epochs 300 --lr-epochs 100 --learning_rate_list '$LEARNING_RATE_LIST
CANDIDATES_PRUNING_RATIOS='0 0 0 0'
MY_DEBUG='--debug 1' # -1: none, 0: info, 1: debug
PRUNE_COMMON_FLAGS='--prune '$PRUNE_METHOD' --sa '$MY_DEBUG
'''
# original training -- 85.26%  --- 87.65%
python main.py $COMMON_FLAGS
# 25% pruned -- 85.56/85.36    --- 86.87/86.42
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 0 \
	--pretrained saved_models/$Model.origin.pth.tar
'''
# 43.1% pruned -- 85.03/85.30  --- /86.34
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 1 \
	--pretrained saved_models/$PRUNE_METHOD/$Model/stage_0.pth.tar
'''
# 57.4% pruned -- /    ---       /85.36
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 2 \
	--pretrained saved_models/$PRUNE_METHOD/$Model/stage_1.pth.tar
# 67.6% pruned -- 78.92/78.88 ---  /83.33
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 3 \
	--pretrained saved_models/$PRUNE_METHOD/$Model/stage_2.pth.tar
# 75.2% pruned -- 78.20(intermittent)/77.83(energy) --- /81.35
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 4 \
	--pretrained saved_models/$PRUNE_METHOD/$Model/stage_3.pth.tar
# 81.1% pruned -- 98.91/99.04   ---   /81.39
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 5 \
	--pretrained saved_models/$PRUNE_METHOD/$Model/stage_4.pth.tar \
	--retrain
# xx% pruned -- 98.85/98.92(energy)  ---  / 80.84
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 6 \
	--pretrained saved_models/$PRUNE_METHOD/$Model/stage_5.pth.tar
# xx% pruned -- 98./98.91(energy) ---  /79.86
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 7 \
	--pretrained saved_models/$PRUNE_METHOD/$Model/stage_6.pth.tar
# xx% pruned -- 98.80/98.66   ---   /77.79
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 8 \
	--pretrained saved_models/$PRUNE_METHOD/$Model/stage_7.pth.tar
