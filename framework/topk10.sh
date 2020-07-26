#!/bin/sh
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 12 --bptt 36 --dropout 0.6482 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 32 --compressor topk --memory residual --exp_name topk_0.1_BS_32 --project_name batch_effect --compress_ratio 0.1

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 17 --bptt 36 --dropout 0.6482 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 64 --compressor topk --memory residual --exp_name topk_0.1_BS_64 --project_name batch_effect --compress_ratio 0.1

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 24 --bptt 36 --dropout 0.6482 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 128 --compressor topk --memory residual --exp_name topk_0.1_BS_128 --project_name batch_effect --compress_ratio 0.1

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 33.9 --bptt 36 --dropout 0.6482 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 256 --compressor topk --memory residual --exp_name topk_0.1_BS_256 --project_name batch_effect --compress_ratio 0.1

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 48  --bptt 36 --dropout 0.6482 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 512 --compressor topk --memory residual --exp_name topk_0.1_BS_512 --project_name batch_effect --compress_ratio 0.1

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 67.9 --bptt 36 --dropout 0.6482 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 1024 --compressor topk --memory residual --exp_name topk_0.1_BS_1024 --project_name batch_effect --compress_ratio 0.1
