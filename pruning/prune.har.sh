Model='HAR'
LEARNING_RATE_LIST='0.0001'
PRUNE_METHOD='' # intermittent or energy
SENA='OFF'
GPUS='0'
VISIBLE_GPUS='0'
OVERALL_PRUNING_RATIO='0.2'
STAGE='0'
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

COMMON_FLAGS='--arch '$Model' --batch-size 600 --test-batch-size 600 --lr 0.0001 --epochs 500 --lr-epochs 100 --visible-gpus '$VISIBLE_GPUS' --gpus '$GPUS' --learning_rate_list '$LEARNING_RATE_LIST
CANDIDATES_PRUNING_RATIOS='0 0 0 0'
MY_DEBUG='--debug 1' # -1: none, 0: info, 1: debug
PRUNE_COMMON_FLAGS='--prune '$PRUNE_METHOD' --sa '$MY_DEBUG' --overall-pruning-ratio '$OVERALL_PRUNING_RATIO
SENSITIVITY_ANALYSIS_FLAGS='--arch '$Model' --batch-size 600 --test-batch-size 600 --lr 0.0001 --epochs 200 --lr-epochs 200 --visible-gpus '$VISIBLE_GPUS' --gpus '$GPUS' --learning_rate_list '$LEARNING_RATE_LIST' --prune '$PRUNE_METHOD' --sen-ana' 

if [[ $PRUNE_METHOD == '' ]]; then
	python3.7 main.py $COMMON_FLAGS
elif [[ $STAGE == '0' ]]; then
	if [[ $SENA = 'ON' ]]; then
		# sensitivity analysis
		python3.7 main.py $SENSITIVITY_ANALYSIS_FLAGS \
			--candidates-pruning-ratios $CANDIDATES_PRUNING_RATIOS \
			--stage 0 \
			--pretrained saved_models/$Model.origin.pth.tar
	else
		python3.7 main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
			--candidates-pruning-ratios $CANDIDATES_PRUNING_RATIOS \
			--stage 0 \
			--pretrained saved_models/$Model.origin.pth.tar
	fi
else
	if [[ $SENA = 'ON' ]]; then
		# sensitivity analysis
		python3.7 main.py $SENSITIVITY_ANALYSIS_FLAGS \
			--candidates-pruning-ratios $CANDIDATES_PRUNING_RATIOS \
			--stage $STAGE \
			--pretrained saved_models/$PRUNE_METHOD/$Model/stage_$(($STAGE - 1)).pth.tar
	else
		python3.7 main.py $COMMON_FLAGS $PRUNE_COMMON_FLAGS \
			--candidates-pruning-ratios $CANDIDATES_PRUNING_RATIOS \
			--stage $STAGE \
			--pretrained saved_models/$PRUNE_METHOD/$Model/stage_$(($STAGE - 1)).pth.tar
	fi
fi
