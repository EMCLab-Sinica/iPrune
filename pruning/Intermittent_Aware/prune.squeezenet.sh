Model='SqueezeNet'
LEARNING_RATE_LIST='0.001 0.0005'
PRUNE_METHOD='' # intermittent or energy
SENA='OFF'
GPUS='0'
VISIBLE_GPUS='2'
OVERALL_PRUNING_RATIO='0.2'
STAGE=''
while getopts a:v:o:sie flag;
do
    case "${flag}" in
        e) PRUNE_METHOD='energy';;
        i) PRUNE_METHOD='intermittent';;
        s) SENA='ON';;
	a) STAGE=${OPTARG};;
        v) VISIBLE_GPUS=${OPTARG};;
        o) OVERALL_PRUNING_RATIO=${OPTARG};;
    esac
done

echo ""
echo "==> Dump Arguments:"
echo "pruning method: "$PRUNE_METHOD;
echo "visible gpus: "$VISIBLE_GPUS;
echo "sensitivity analysis: "$SENA;
echo "overall pruning ratio: "$OVERALL_PRUNING_RATIO;
echo "stage: "$STAGE;
echo ""

COMMON_FLAGS='--arch '$Model' --batch-size 8 --test-batch-size 8 --lr 0.0001 --epochs 300 --lr-epochs 150 --visible-gpus '$VISIBLE_GPUS' --gpus '$GPUS' --learning_rate_list '$LEARNING_RATE_LIST
CANDIDATES_PRUNING_RATIOS='0 0 0 0 0 0 0 0 0 0 0'
MY_DEBUG='--debug -1'
PRUNE_COMMON_FLAGS='--prune '$PRUNE_METHOD' --sa '$MY_DEBUG' --overall-pruning-ratio '$OVERALL_PRUNING_RATIO
SENSITIVITY_ANALYSIS_FLAGS='--arch '$Model' --batch-size 256 --test-batch-size 256 --lr 0.0005 --epochs 100 --lr-epochs 50 --visible-gpus '$VISIBLE_GPUS' --gpus '$GPUS' --learning_rate_list '$LEARNING_RATE_LIST' --prune '$PRUNE_METHOD' --sen-ana'

"""
origin: xx%
|stage|intermittent prune|energy prune|
|0|||
|1|||
|2|||
|3|||
|4|||
"""

if [ $PRUNE_METHOD == '' ]; then
	python main.py $COMMON_FLAGS
elif [ $STAGE == '0' ]; then
	# sensitivity analysis
	python main.py $SENSITIVITY_ANALYSIS_FLAGS \
		--candidates-pruning-ratios $CANDIDATES_PRUNING_RATIOS \
		--stage 0 \
		--pretrained saved_models/$Model.origin.pth.tar
	python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
		--stage 0 \
		--pretrained saved_models/$Model.origin.pth.tar
else
	if [ $SENA = 'ON' ]; then
		# sensitivity analysis
		python main.py $SENSITIVITY_ANALYSIS_FLAGS \
			--candidates-pruning-ratios $CANDIDATES_PRUNING_RATIOS \
			--stage $STAGE \
			--pretrained saved_models/$PRUNE_METHOD/$Model/stage_$(($STAGE - 1)).pth.tar
	else
		python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
			--stage $STAGE \
			--pretrained saved_models/$PRUNE_METHOD/$Model/stage_$(($STAGE - 1)).pth.tar
	fi
fi

'''
# original training --75.76%
python main.py $COMMON_FLAGS
# sensitivity analysis
python main.py $SENSITIVITY_ANALYSIS_FLAGS \
	--candidates-pruning-ratios $CANDIDATES_PRUNING_RATIOS \
	--stage 0 \
	--pretrained saved_models/$Model.origin.pth.tar
# 76.65%/76.48%
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 0 \
	--pretrained saved_models/$Model.origin.pth.tar \
	--candidates-pruning-ratios $CANDIDATES_PRUNING_RATIOS
# 76.75/77.13%
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 1 \
	--pretrained saved_models/$PRUNE_METHOD/$Model/stage_0.pth.tar \
	--candidates-pruning-ratios $CANDIDATES_PRUNING_RATIOS
# 76.33/76.48
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 2 \
	--pretrained saved_models/$PRUNE_METHOD/$Model/stage_1.pth.tar \
	--candidates-pruning-ratios $CANDIDATES_PRUNING_RATIOS
# 75.59 (intermittent)/75.35(energy)
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 3 \
	--pretrained saved_models/$PRUNE_METHOD/$Model/stage_2.pth.tar \
	--candidates-pruning-ratios $CANDIDATES_PRUNING_RATIOS
# xxx/xxx
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 4 \
	--pretrained saved_models/$PRUNE_METHOD/$Model/stage_3.pth.tar \
	--candidates-pruning-ratios $CANDIDATES_PRUNING_RATIOS
# xxx/xxx
python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
	--stage 5 \
	--pretrained saved_models/$PRUNE_METHOD/$Model/stage_4.pth.tar \
	--candidates-pruning-ratios $CANDIDATES_PRUNING_RATIOS
