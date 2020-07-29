#!/bin/sh
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.25 --bptt 29 --dropout 0.6904 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 32 --compressor efsignsgd --memory residual --exp_name EF_SignSGD_BS_32_k --project_name batch_effect

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 10.5 --bptt 29 --dropout 0.6904 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 64 --compressor efsignsgd --memory residual --exp_name EF_SignSGD_BS_64_k --project_name batch_effect

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 21.0 --bptt 29 --dropout 0.6904 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 128 --compressor efsignsgd --memory residual --exp_name EF_SignSGD_BS_128_k --project_name batch_effect

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 42 --bptt 29 --dropout 0.6904 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 256 --compressor efsignsgd --memory residual --exp_name EF_SignSGD_BS_256_k --project_name batch_effect

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 84 --bptt 29 --dropout 0.6904 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 512 --compressor efsignsgd --memory residual --exp_name EF_SignSGD_BS_512_k --project_name batch_effect

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 168 --bptt 29 --dropout 0.6904 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 1024 --compressor efsignsgd --memory residual --exp_name EF_SignSGD_BS_1024_k --project_name batch_effect
