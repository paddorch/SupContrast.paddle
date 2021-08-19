set -ex

export CUDA_VISIBLE_DEVICES=0
mkdir -p output
#export FLAGS_enable_parallel_graph=1
#export FLAGS_sync_nccl_allreduce=1
nohup python main_supcon.py -y yamls/resnet50_supcon.yml > output/supcon.out 2>&1 &
#python main_supcon.py -y yamls/resnet50_supcon.yml
