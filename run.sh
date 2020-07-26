#TopK_0.1
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 24.0 --bptt 36 --dropout 0.6482 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 128 --compressor topk --compress_ratio 0.1 --memory residual --exp_name TopK_0.1_1_worker --project_name multi_workers_1

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 24.0 --bptt 36 --dropout 0.6482 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 2 --batch_size 64 --compressor topk --compress_ratio 0.1 --memory residual --exp_name TopK_0.1_2_workers --project_name multi_workers_1

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 24.0 --bptt 36 --dropout 0.6482 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 4 --batch_size 32 --compressor topk --compress_ratio 0.1 --memory residual --exp_name TopK_0.1_4_workers --project_name multi_workers_1

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 24.0 --bptt 36 --dropout 0.6482 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 8 --batch_size 16 --compressor topk --compress_ratio 0.1 --memory residual --exp_name TopK_0.1_8_workers --project_name multi_workers_1


#TopK_0.01
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 20.0 --bptt 43 --dropout 0.7003 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 128 --compressor topk --compress_ratio 0.01 --memory residual --exp_name TopK_0.01_1_worker --project_name multi_workers_1

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 20.0 --bptt 43 --dropout 0.7003 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 2 --batch_size 64 --compressor topk --compress_ratio 0.01 --memory residual --exp_name TopK_0.01_2_workers --project_name multi_workers_1

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 20.0 --bptt 43 --dropout 0.7003 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 4 --batch_size 32 --compressor topk --compress_ratio 0.01 --memory residual --exp_name TopK_0.01_4_workers --project_name multi_workers_1

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 20.0 --bptt 43 --dropout 0.7003 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 8 --batch_size 16 --compressor topk --compress_ratio 0.01 --memory residual --exp_name TopK_0.01_8_workers --project_name multi_workers_1


#TopK_0.01
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 20.0 --bptt 43 --dropout 0.7003 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 128 --compressor topk --compress_ratio 0.001 --memory residual --exp_name TopK_0.001_1_worker --project_name multi_workers_1

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 20.0 --bptt 43 --dropout 0.7003 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 2 --batch_size 64 --compressor topk --compress_ratio 0.001 --memory residual --exp_name TopK_0.001_2_workers --project_name multi_workers_1

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 20.0 --bptt 43 --dropout 0.7003 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 4 --batch_size 32 --compressor topk --compress_ratio 0.001 --memory residual --exp_name TopK_0.001_4_workers --project_name multi_workers_1

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 20.0 --bptt 43 --dropout 0.7003 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 8 --batch_size 16 --compressor topk --compress_ratio 0.001 --memory residual --exp_name TopK_0.001_8_workers --project_name multi_workers_1


#TernGrad
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --bptt 20 --dropout 0.4854 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 128 --compressor terngrad --memory none --clip_const 25.673 --exp_name TernGrad_1_worker --project_name multi_workers_1

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --bptt 20 --dropout 0.4854 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 2 --batch_size 64 --compressor terngrad --memory none --clip_const 25.673 --exp_name TernGrad_2_workers --project_name multi_workers_1

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --bptt 20 --dropout 0.4854 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 4 --batch_size 32 --compressor terngrad --memory none --clip_const 25.673 --exp_name TernGrad_4_workers --project_name multi_workers_1

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --bptt 20 --dropout 0.4854 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 8 --batch_size 16 --compressor terngrad --memory none --clip_const 25.673 --exp_name TernGrad_8_workers --project_name multi_workers_1


#TernGrad
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 7.0 --bptt 15 --dropout 0.4928 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 128 --compressor adacomp --memory residual --comp_const 2.35 --exp_name AdaComp_1_worker --project_name multi_workers_1

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 7.0 --bptt 15 --dropout 0.4928 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 2 --batch_size 64 --compressor adacomp --memory residual --comp_const 2.35 --exp_name AdaComp_2_workers --project_name multi_workers_1

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 7.0 --bptt 15 --dropout 0.4928 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 4 --batch_size 32 --compressor adacomp --memory residual --comp_const 2.35 --exp_name AdaComp_4_workers --project_name multi_workers_1

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 7.0 --bptt 15 --dropout 0.4928 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 8 --batch_size 16 --compressor adacomp --memory residual --comp_const 2.35 --exp_name AdaComp_8_workers --project_name multi_workers_1
