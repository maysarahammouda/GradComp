#!/bin/sh
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 20 --bptt 27 --dropout 0.6429 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 512 --compressor none --memory none --exp_name no_compression_BS_512_20 --project_name batch_effect2

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 28.3 --bptt 27 --dropout 0.6429 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 512 --compressor none --memory none --exp_name no_compression_BS_512_28.3 --project_name batch_effect2


python main.py --data ../datasets/ptb --num_layers 2 --init_lr 24 --bptt 36 --dropout 0.6482 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 512 --compressor topk --memory residual --exp_name topk_0.1_BS_512_24 --project_name batch_effect2 --compress_ratio 0.1

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 29.7 --bptt 36 --dropout 0.6482 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 512 --compressor topk --memory residual --exp_name topk_0.1_BS_512_29.7 --project_name batch_effect2 --compress_ratio 0.1


python main.py --data ../datasets/ptb --num_layers 2 --init_lr 20 --bptt 43 --dropout 0.7003 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 512 --compressor topk --memory residual --exp_name topk_0.01_BS_512_20 --project_name batch_effect2 --compress_ratio 0.01

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 28.3 --bptt 43 --dropout 0.7003 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 512 --compressor topk --memory residual --exp_name topk_0.01_BS_512_28.3 --project_name batch_effect2 --compress_ratio 0.01


python main.py --data ../datasets/ptb --num_layers 2 --init_lr 20 --bptt 43 --dropout 0.7003 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 512 --compressor topk --memory residual --exp_name topk_0.001_BS_512_20 --project_name batch_effect2 --compress_ratio 0.001

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 28.3 --bptt 43 --dropout 0.7003 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 512 --compressor topk --memory residual --exp_name topk_0.001_BS_512_28.3 --project_name batch_effect2 --compress_ratio 0.001


python main.py --data ../datasets/ptb --num_layers 2 --init_lr 20 --bptt 43 --dropout 0.7003 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 256 --compressor topk --memory residual --exp_name topk_0.001_BS_256_20 --project_name batch_effect2 --compress_ratio 0.001

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 20 --bptt 27 --dropout 0.6429 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 256 --compressor none --memory none --exp_name no_compression_BS_256_20 --project_name batch_effect2

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 20 --bptt 43 --dropout 0.7003 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 256 --compressor topk --memory residual --exp_name topk_0.01_BS_256_20 --project_name batch_effect2 --compress_ratio 0.01
