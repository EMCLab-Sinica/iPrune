Model='HAR'
PRUNE_METHOD=$1
COMMON_FLAGS='--arch '$Model' --batch-size 600 --test-batch-size 600 --lr 0.0001 --epochs 500'
CANDIDATES_PRUNING_RATIOS='0 0 0 0'
MY_DEBUG='--debug 1' # -1: none, 0: info, 1: debug
PRUNE_COMMON_FLAGS='--prune '$PRUNE_METHOD' --sa '$MY_DEBUG' --overall-pruning-ratio 0.2'
SENSITIVITY_ANALYSIS_FLAGS='--arch '$Model' --batch-size 600 --test-batch-size 600 --lr 0.0001 --epochs 200 --prune '$PRUNE_METHOD' --sen-ana'
			    # intermittent/energy
# original training -- 92.50%
python main.py $COMMON_FLAGS
'''
# sensitivity analysis
python main.py $SENSITIVITY_ANALYSIS_FLAGS \
	--candidates-pruning-ratios $CANDIDATES_PRUNING_RATIOS \
	--stage 0 \
	--pretrained saved_models/HAR.origin.pth.tar
# 25% pruned -- 92.16/92.20
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 0 \
	--pretrained saved_models/HAR.origin.pth.tar
# 92.37/92.67
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 1 \
	--pretrained saved_models/$PRUNE_METHOD/$Model/stage_0.pth.tar
# 92.26 (intermittent)/92.03
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 2 \
	--pretrained saved_models/$PRUNE_METHOD/$Model/stage_1.pth.tar
# 92.03/92.38
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 3 \
	--pretrained saved_models/$PRUNE_METHOD/$Model/stage_2.pth.tar
# xxxx/92.18 (energy)
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 4 \
	--pretrained saved_models/$PRUNE_METHOD/$Model/stage_3.pth.tar
# xxxx/91.86
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 5 \
	--pretrained saved_models/$PRUNE_METHOD/$Model/stage_4.pth.tar
