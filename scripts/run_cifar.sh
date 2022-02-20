NUM_RUNS=1
GPU_IDS=( 3 )
NUM_GPUS=${#GPU_IDS[@]}
counter=0

SEED=( 42 ) 
LR=( 0.1 )
DATATYPE=( 'cifar_DogCat' )
TRAINMETHOD=( 'PvU' )
NETTYPE=( 'ResNet' )
ALPHA=( 0.5 )

for seed in "${SEED[@]}"; do 
for alpha in "${ALPHA[@]}"; do
for lr in "${LR[@]}"; do
for datatype in "${DATATYPE[@]}"; do
for trainmethod in "${TRAINMETHOD[@]}"; do
for nettype in "${NETTYPE[@]}"; do
	 # Get GPU id.
	 gpu_idx=$((counter/2 % $NUM_GPUS))
	 gpu_id=${GPU_IDS[$gpu_idx]}

	 counter=$((counter+1))
	 

	 if [ "$trainmethod" = "nnPU" ] || [ "$trainmethod" = "uPU" ]; then 
	 	cmd="CUDA_VISIBLE_DEVICES=${gpu_id} python train_PU.py --lr=0.0001 --momentum=0.0\
      		--data-type=${datatype} --train-method=${trainmethod} --net-type=${nettype} --epochs=2000 --optimizer=Adam --alpha=${alpha} --seed=${seed}"
	 else
	 	cmd="CUDA_VISIBLE_DEVICES=${gpu_id} python train_PU.py --lr=${lr} --momentum=0.9\
      		--data-type=${datatype} --train-method=${trainmethod} --net-type=${nettype} --epochs=2000  --seed=${seed} --alpha=${alpha} --warm-start --warm-start-epochs=100"
      	 fi 

         echo $cmd
 	 echo $count $of
	 eval ${cmd} &

	 if ! ((counter % NUM_RUNS)); then
		  wait
	 fi
done
done
done
done
done
done
