NUM_RUNS=4
GPU_IDS=( 0 1 2 3 )
NUM_GPUS=${#GPU_IDS[@]}
counter=0

SEED=( 42 1432 8378 )
LR=( 0.00005 )
DATATYPE=( 'IMDb_BERT' )
TRAINMETHOD=( 'PvU' 'CVIR' 'nnPU' 'uPU' 'TEDn' )
NETTYPE=( 'DistilBert' )
ALPHA=( 0.5 )

for seed in "${SEED[@]}"; do
for alpha in "${ALPHA[@]}"; do
for lr in "${LR[@]}"; do
for datatype in "${DATATYPE[@]}"; do
for nettype in "${NETTYPE[@]}"; do
for trainmethod in "${TRAINMETHOD[@]}"; do
	 # Get GPU id.
	 gpu_idx=$((counter % $NUM_GPUS))
	 gpu_id=${GPU_IDS[$gpu_idx]}

	 if [ "$trainmethod" = "nn_unbiased" ] || [ "$trainmethod" = "unbiased" ]; then
	 	cmd="CUDA_VISIBLE_DEVICES=${gpu_id} python train_PU.py --lr=0.00001 --momentum=0.0\
      		--data-type=${datatype} --train-method=${trainmethod} --net-type=${nettype}  --alpha=${alpha}  --epochs=50 --optimizer=AdamW"
	 else
	 	cmd="CUDA_VISIBLE_DEVICES=${gpu_id} python train_PU.py --lr=${lr} --momentum=0.0\
      		--data-type=${datatype} --train-method=${trainmethod} --net-type=${nettype} --epochs=50  --optimizer=AdamW --alpha=${alpha}  --warm-start --warm-start-epochs=2"
      	 fi 

         echo $cmd
 	 echo $count $of
	 eval ${cmd} &

	 counter=$((counter+1))
	 if ! ((counter % NUM_RUNS)); then
		  wait
	 fi
done
done
done
done
done
done
