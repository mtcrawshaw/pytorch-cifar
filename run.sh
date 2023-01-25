#!/bin/bash
start_gpu=$1
dataset=$2

if [[ $# -ne 2 ]]; then
    echo "Usage: bash run.sh start_gpu dataset"
    exit
fi

if [[ "$dataset" == "CIFAR10" ]]; then
    epochs=200
elif [[ "$dataset" == "MNIST" ]]; then
    epochs=10
else
    echo "Unrecognized dataset: $dataset."
    exit
fi

mkdir -p logs

python3 main.py --gpu $((${start_gpu})) --optimizer SGD --lr 0.1 --dataset $dataset --epochs $epochs --name ${dataset}_sgd > logs/${dataset}_sgd_out.txt &
python3 main.py --gpu $((${start_gpu}+1)) --optimizer Adam --lr 0.1 --beta1 0.9 --beta2 0.999 --eps 1e-8 --dataset $dataset --epochs $epochs --name ${dataset}_adam > logs/${dataset}_adam_out.txt &
python3 main.py --gpu $((${start_gpu}+2)) --optimizer Adam --lr 0.1 --beta2 0.9 --beta2 0.0 --eps 0.1 --dataset $dataset --epochs $epochs --name ${dataset}_adam_nobeta2 > logs/${dataset}_adam_nobeta2_out.txt &
wait
