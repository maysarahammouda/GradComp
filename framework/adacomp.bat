#Different batch sizes - AdaComb
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 3.5 --bptt 15 --dropout 0.4928 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 32 --compressor adacomp --memory residual --exp_name adacomp_BS_32 --project_name batch_effect --comp_const 2.35

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 4.9 --bptt 15 --dropout 0.4928 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 64 --compressor adacomp --memory residual --exp_name adacomp_BS_64 --project_name batch_effect --comp_const 2.35

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 7.0 --bptt 15 --dropout 0.4928 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 128 --compressor adacomp --memory residual --exp_name adacomp_BS_128 --project_name batch_effect --comp_const 2.35

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 9.9 --bptt 15 --dropout 0.4928 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 256 --compressor adacomp --memory residual --exp_name adacomp_BS_256 --project_name batch_effect --comp_const 2.35

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 14.0 --bptt 15 --dropout 0.4928 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 512 --compressor adacomp --memory residual --exp_name adacomp_BS_512 --project_name batch_effect --comp_const 2.35

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 19.8 --bptt 15 --dropout 0.4928 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 1024 --compressor adacomp --memory residual --exp_name adacomp_BS_1024 --project_name batch_effect --comp_const 2.35
