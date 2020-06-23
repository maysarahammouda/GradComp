#1_worker
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 1 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 1 --batch_size 128 --exp_name 1_worker --project_name bs128_nhid650

#2_workers
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 1 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 2 --batch_size 64 --exp_name 2_workers --project_name bs128_nhid650

#4_workers
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 1 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 4 --batch_size 32 --exp_name 4_workers --project_name bs128_nhid650

#8_workers
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 1 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 8 --batch_size 16 --exp_name 8_workers --project_name bs128_nhid650

#16_workers
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 1 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 16 --batch_size 8 --exp_name 16_workers --project_name bs128_nhid650


#1_worker(test)
python main.py --data ../datasets/test --num_layers 2 --init_lr 5.0 --epochs 1 --eval_batch_size 5 --test_batch_size 5 --bptt 2 --dropout 1 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu false --emb_size 20 --num_hid 20 --num_workers 1 --batch_size 8 --exp_name 1_worker_test --project_name test --compress_ratio 1

python main.py --data ../datasets/test --num_layers 2 --init_lr 5.0 --epochs 1 --eval_batch_size 5 --test_batch_size 5 --bptt 2 --dropout 1 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu false --emb_size 20 --num_hid 20 --num_workers 1 --batch_size 8 --exp_name 1_worker_test --project_name test --compress_ratio 1
