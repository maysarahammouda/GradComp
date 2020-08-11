#!/bin/sh
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 4 --bptt 20 --dropout 0.5729 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 32 --compressor adacomp --memory residual --exp_name adacomp_BS_32 --project_name batch_effect2 --comp_const 2.35

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.7 --bptt 20 --dropout 0.5729 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 64 --compressor adacomp --memory residual --exp_name adacomp_BS_64 --project_name batch_effect2 --comp_const 2.35

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 8 --bptt 20 --dropout 0.5729 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 128 --compressor adacomp --memory residual --exp_name adacomp_BS_128 --project_name batch_effect2 --comp_const 2.35

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 11.3 --bptt 20 --dropout 0.5729 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 256 --compressor adacomp --memory residual --exp_name adacomp_BS_256 --project_name batch_effect2 --comp_const 2.35

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 16 --bptt 20 --dropout 0.5729 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 512 --compressor adacomp --memory residual --exp_name adacomp_BS_512 --project_name batch_effect2 --comp_const 2.35

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 11.3 --bptt 20 --dropout 0.5729 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 512 --compressor adacomp --memory residual --exp_name adacomp_BS_512_11.3 --project_name batch_effect2 --comp_const 2.35

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 8 --bptt 20 --dropout 0.5729 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 512 --compressor adacomp --memory residual --exp_name adacomp_BS_512_8 --project_name batch_effect2 --comp_const 2.35

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 8 --bptt 20 --dropout 0.5729 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 256 --compressor adacomp --memory residual --exp_name adacomp_BS_256_8 --project_name batch_effect2 --comp_const 2.35
