#! /bin/bash
environs="WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1"

# 运行命令
run_cmd="$environs python -m torch.distributed.run \
    --nproc_per_node=1 \
    mtv/app.py --base ./configs/cogvideox_5b.yaml ./configs/inference.yaml \
    --seed $RANDOM --output-dir $1"

echo ${run_cmd}
eval ${run_cmd}

echo "DONE on `hostname`"
