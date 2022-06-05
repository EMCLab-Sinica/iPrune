Model='KWS_CNN_S'
LEARNING_RATE_LIST='0.0005 0.0001 0.00002'
PRUNE_METHOD='' # intermittent or energy
SENA='OFF'
GPUS='0'
VISIBLE_GPUS='4'
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

COMMON_FLAGS='--arch '$Model' --batch-size 16 --test-batch-size 16 --lr 0.0005 --epochs 150 --lr-epochs 50 --visible-gpus '$VISIBLE_GPUS' --gpus '$GPUS' --learning_rate_list '$LEARNING_RATE_LIST
CANDIDATES_PRUNING_RATIOS='0 0 0 0 0'
MY_DEBUG='--debug 1' # -1: none, 0: info, 1: debug
PRUNE_COMMON_FLAGS='--prune '$PRUNE_METHOD' --sa '$MY_DEBUG' --overall-pruning-ratio '$OVERALL_PRUNING_RATIO
SENSITIVITY_ANALYSIS_FLAGS='--arch '$Model' --batch-size 512 --test-batch-size 512 --lr 0.0005 --epochs 50 --lr-epochs 100 --visible-gpus '$VISIBLE_GPUS' --gpus '$GPUS' --learning_rate_list '$LEARNING_RATE_LIST' --prune '$PRUNE_METHOD' --sen-ana'

"""
origin: 88.23%
|stage|intermittent prune|energy prune|
|0|87.93|88.04|
|1|87.73|87.32|
|2|87.38|87.46|
|3|85.95|85.85|
|4|84.76|85.54|
"""

if [[ $PRUNE_METHOD == '' ]]; then
	python main.py $COMMON_FLAGS
elif [[ $STAGE == '0' ]]; then
	if [[ $SENA = 'ON' ]]; then
		# sensitivity analysis
		python main.py $SENSITIVITY_ANALYSIS_FLAGS \
			--candidates-pruning-ratios $CANDIDATES_PRUNING_RATIOS \
			--stage 0 \
			--pretrained saved_models/$Model.origin.pth.tar
	else
		python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
			--stage 0 \
			--pretrained saved_models/$Model.origin.pth.tar
	fi
else
	if [[ $SENA = 'ON' ]]; then
		# sensitivity analysis
		python main.py $SENSITIVITY_ANALYSIS_FLAGS \
			--candidates-pruning-ratios $CANDIDATES_PRUNING_RATIOS \
			--stage $STAGE \
			--pretrained saved_models/$PRUNE_METHOD/$Model/stage_$(($STAGE - 1)).pth.tar
	else
		python main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
			--candidates-pruning-ratios $CANDIDATES_PRUNING_RATIOS \
			--stage $STAGE \
			--pretrained saved_models/$PRUNE_METHOD/$Model/stage_$(($STAGE - 1)).pth.tar
	fi
fi
