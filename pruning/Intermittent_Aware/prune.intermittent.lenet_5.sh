COMMON_FLAGS='--arch LeNet_5'
GROUP_SIZE='1 1 5 5'

# original training -- 99.34%
# python main.py $COMMON_FLAGS

# 83.4% pruned / 92.6% pruned (NCHW) -- 99.18%
python main.py $COMMON_FLAGS --prune 'intermittent' --stage 0 \
 	--group $GROUP_SIZE \
	--pretrained saved_models/LeNet_5.new.pad.best_origin.pth.tar \
	--lr 0.1 --lr-epochs 15 --threshold 0.04

# 92.1% pruned / 93.7% pruned (NHWC) -- 99.11%
# python main.py $COMMON_FLAGS --stage 1 --prune 'intermittent' --group $GROUP_SIZE\
#	--pretrained saved_models/LeNet_5.intermittent.simd.0.pth.tar \
#	--lr 0.01 --lr-epochs 20 --threshold 0.05

# 93.6% pruned / 95.2% pruned (NHWC) -- 99.13%
# python main.py $COMMON_FLAGS --prune simd --stage 2 --width $SIMD_WIDTH\
#	--pretrained saved_models/LeNet_5.prune.simd.1.pth.tar \
#	--lr 0.01 --lr-epochs 20 --threshold 0.06

# 95.9% pruned / 96.6% pruned (NHWC) -- 99.14%
# python main.py $COMMON_FLAGS --prune simd --stage 3 --width $SIMD_WIDTH\
#	--pretrained saved_models/LeNet_5.prune.simd.2.pth.tar \
#	--lr 0.01 --lr-epochs 20 --threshold 0.075

# 96.8% pruned / 97.2% pruned (NHWC) -- 99.10%
# python main.py $COMMON_FLAGS --prune simd --stage 4 --width $SIMD_WIDTH\
#	--pretrained saved_models/LeNet_5.prune.simd.3.pth.tar \
#	--lr 0.01 --lr-epochs 20 --threshold 0.080

# 			   / 99.1% pruned (NHWC) -- 98.95%
# python main.py $COMMON_FLAGS --prune simd --stage 5 --width $SIMD_WIDTH\
#	--pretrained saved_models/LeNet_5.prune.simd.4.pth.tar \
#	--lr 0.01 --lr-epochs 20 --threshold 0.140
