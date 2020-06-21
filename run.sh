#exp1_worker1
python main.py --data ../datasets/ptb --nlayers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 1 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --embsize 650 --nhid 650 --nworker 1 --batch_size 128 --exp_name exp1_worker1 --project_name bs128_nhid650

#exp1_worker2
python main.py --data ../datasets/ptb --nlayers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 1 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --embsize 650 --nhid 650 --nworker 2 --batch_size 64 --exp_name exp1_worker2 --project_name bs128_nhid650

#exp1_worker4
python main.py --data ../datasets/ptb --nlayers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 1 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --embsize 650 --nhid 650 --nworker 4 --batch_size 32 --exp_name exp1_worker4 --project_name bs128_nhid650

#exp1_worker8
python main.py --data ../datasets/ptb --nlayers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 1 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --embsize 650 --nhid 650 --nworker 8 --batch_size 16 --exp_name exp1_worker8 --project_name bs128_nhid650

#exp1_worker16
python main.py --data ../datasets/ptb --nlayers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 1 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --embsize 650 --nhid 650 --nworker 16 --batch_size 8 --exp_name exp1_worker16 --project_name bs128_nhid650
