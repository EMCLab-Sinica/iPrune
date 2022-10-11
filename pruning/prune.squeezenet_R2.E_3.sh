Model='SqueezeNet'
LEARNING_RATE_LIST='0.001 0.0005'
PRUNE_METHOD='energy' # intermittent or energy
SENA='OFF'
GPUS='3'
VISIBLE_GPUS='4'
OVERALL_PRUNING_RATIO='0.2'
STAGE='3'
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

COMMON_FLAGS='--arch '$Model' --batch-size 128 --test-batch-size 128 --lr 0.0001 --epochs 150 --lr-epochs 50 --visible-gpus '$VISIBLE_GPUS' --gpus '$GPUS' --learning_rate_list '$LEARNING_RATE_LIST
CANDIDATES_PRUNING_RATIOS='0 0 0 0 0 0 0 0 0 0 0'
MY_DEBUG='--debug -1'
PRUNE_COMMON_FLAGS='--prune '$PRUNE_METHOD' --sa '$MY_DEBUG' --overall-pruning-ratio '$OVERALL_PRUNING_RATIO
SENSITIVITY_ANALYSIS_FLAGS='--arch '$Model' --batch-size 256 --test-batch-size 256 --lr 0.0005 --epochs 150 --lr-epochs 75 --visible-gpus '$VISIBLE_GPUS' --gpus '$GPUS' --learning_rate_list '$LEARNING_RATE_LIST' --prune '$PRUNE_METHOD' --sen-ana'

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
			--candidates-pruning-ratios $CANDIDATES_PRUNING_RATIOS \
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
