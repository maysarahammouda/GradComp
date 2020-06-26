python main.py --data ../datasets/test --num_layers 2 --init_lr 5.0 --epochs 3 --eval_batch_size 5 --test_batch_size 5 --bptt 2 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu false --emb_size 20 --num_hid 20 --num_workers 1 --batch_size 8 --compressor topk --memory residual --exp_name test --project_name test

#1_worker
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 1 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --nhid 650 --num_workers 1 --batch_size 128 --exp_name exp1_worker1 --project_name bs128_nhid650

#2_worker
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 1 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --nhid 650 --num_workers 2 --batch_size 64 --exp_name exp1_worker2 --project_name bs128_nhid650

#4_worker
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 1 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --nhid 650 --num_workers 4 --batch_size 32 --exp_name exp1_worker4 --project_name bs128_nhid650

#8_worker
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 1 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --nhid 650 --num_workers 8 --batch_size 16 --exp_name exp1_worker8 --project_name bs128_nhid650

#1_worker6
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 1 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --nhid 650 --num_workers 16 --batch_size 8 --exp_name exp1_worker16 --project_name bs128_nhid650



#1_workers
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 1 --batch_size 128 --exp_name 1_worker --project_name bs128_nhid650_drop0.5

#2_workers
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 2 --batch_size 64 --exp_name 2_workers --project_name bs128_nhid650_drop0.5

#4_workers
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 4 --batch_size 32 --exp_name 4_workers --project_name bs128_nhid650_drop0.5

#8_workers
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 8 --batch_size 16 --exp_name 8_workers --project_name bs128_nhid650_drop0.5

#16_workers
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 16 --batch_size 8 --exp_name 16_workers --project_name bs128_nhid650_drop0.5

#32_workers
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 32 --batch_size 4 --exp_name 32_workers --project_name bs128_nhid650_drop0.5




#1_worker
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.25 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --nhid 650 --num_workers 1 --batch_size 128 --exp_name 1_worker --project_name bs128_nhid650_drop0.25

#2_worker
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.25 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --nhid 650 --num_workers 2 --batch_size 64 --exp_name 2_workers --project_name bs128_nhid650_drop0.25

#4_worker
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.25 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --nhid 650 --num_workers 4 --batch_size 32 --exp_name 4_workers --project_name bs128_nhid650_drop0.25

#8_worker
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.25 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --nhid 650 --num_workers 8 --batch_size 16 --exp_name 8_workers --project_name bs128_nhid650_drop0.25

#16_worker
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.25 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --nhid 650 --num_workers 16 --batch_size 8 --exp_name 16_workers --project_name bs128_nhid650_drop0.25


#1_worker
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.75 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --nhid 650 --num_workers 1 --batch_size 128 --exp_name 1_worker --project_name bs128_nhid650_drop0.75

#2_workers
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.75 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --nhid 650 --num_workers 2 --batch_size 64 --exp_name 2_workers --project_name bs128_nhid650_drop0.75

#4_workers
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.75 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --nhid 650 --num_workers 4 --batch_size 32 --exp_name 4_workers --project_name bs128_nhid650_drop0.75

#8_workers
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.75 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --nhid 650 --num_workers 8 --batch_size 16 --exp_name 8_workers --project_name bs128_nhid650_drop0.75

#16_workers
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.75 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --nhid 650 --num_workers 16 --batch_size 8 --exp_name 16_workers --project_name bs128_nhid650_drop0.75


#1_worker
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 1000 --nhid 1000 --num_workers 1 --batch_size 128 --exp_name 1_worker --project_name bs128_nhid1000_drop0.5

#2_worker
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 1000 --nhid 1000 --num_workers 2 --batch_size 64 --exp_name 2_workers --project_name bs128_nhid1000_drop0.5

#4_worker
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 1000 --nhid 1000 --num_workers 4 --batch_size 32 --exp_name 4_workers --project_name bs128_nhid1000_drop0.5

#8_worker
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.75 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 1000 --nhid 1000 --num_workers 8 --batch_size 16 --exp_name 8_workers --project_name bs128_nhid1000_drop0.5

#1_worker6
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.75 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 1000 --nhid 1000 --num_workers 16 --batch_size 8 --exp_name 16_workers --project_name bs128_nhid1000_drop0.5




#1_worker
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 500 --nhid 500 --num_workers 1 --batch_size 128 --exp_name 1_worker --project_name bs128_nhid500_drop0.5

#2_worker
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 500 --nhid 500 --num_workers 2 --batch_size 64 --exp_name 2_workers --project_name bs128_nhid500_drop0.5

#4_worker
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 500 --nhid 500 --num_workers 4 --batch_size 32 --exp_name 4_workers --project_name bs128_nhid500_drop0.5

#8_worker
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.75 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 500 --nhid 500 --num_workers 8 --batch_size 16 --exp_name 8_workers --project_name bs128_nhid500_drop0.5

#1_worker6
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.75 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 500 --nhid 500 --num_workers 16 --batch_size 8 --exp_name 16_workers --project_name bs128_nhid500_drop0.5



#1_worker
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 400 --nhid 400 --num_workers 1 --batch_size 128 --exp_name 1_worker --project_name bs128_nhid400_drop0.5

#2_worker
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 400 --nhid 400 --num_workers 2 --batch_size 64 --exp_name 2_workers --project_name bs128_nhid400_drop0.5

#4_worker
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 400 --nhid 400 --num_workers 4 --batch_size 32 --exp_name 4_workers --project_name bs128_nhid400_drop0.5

#8_worker
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.75 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 400 --nhid 400 --num_workers 8 --batch_size 16 --exp_name 8_workers --project_name bs128_nhid400_drop0.5

#1_worker6
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.75 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 400 --nhid 400 --num_workers 16 --batch_size 8 --exp_name 16_workers --project_name bs128_nhid500_drop0.5



#1_worker
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 1 --batch_size 128 --compress_ratio 1 --exp_name no_compression --project_name bs128_nhid650_drop0.5_compress

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 1 --batch_size 128 --compress_ratio 0.1 --exp_name TopK_0.1 --project_name bs128_nhid650_drop0.5_compress
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 1 --batch_size 128 --compress_ratio 0.01 --exp_name TopK_0.01 --project_name bs128_nhid650_drop0.5_compress
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 1 --batch_size 128 --compress_ratio 0.001 --exp_name TopK_0.001 --project_name bs128_nhid650_drop0.5_compress

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 1 --batch_size 128 --compress_ratio 0.1 --exp_name RandomK_0.1 --project_name bs128_nhid650_drop0.5_compress
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 1 --batch_size 128 --compress_ratio 0.01 --exp_name RandomK_0.01 --project_name bs128_nhid650_drop0.5_compress
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 1 --batch_size 128 --compress_ratio 0.001 --exp_name RandomK_0.001 --project_name bs128_nhid650_drop0.5_compress

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 1 --batch_size 128 --compress_ratio 0.1 --exp_name TopK_0.1_ResMem --project_name bs128_nhid650_drop0.5_compress
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 1 --batch_size 128 --compress_ratio 0.01 --exp_name TopK_0.01_ResMem --project_name bs128_nhid650_drop0.5_compress
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 1 --batch_size 128 --compress_ratio 0.001 --exp_name TopK_0.001_ResMem --project_name bs128_nhid650_drop0.5_compress

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 1 --batch_size 128 --compress_ratio 0.1 --exp_name RandomK_0.1_ResMem --project_name bs128_nhid650_drop0.5_compress
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 1 --batch_size 128 --compress_ratio 0.01 --exp_name RandomK_0.01_ResMem --project_name bs128_nhid650_drop0.5_compress
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 1 --batch_size 128 --compress_ratio 0.001 --exp_name RandomK_0.001_ResMem --project_name bs128_nhid650_drop0.5_compress



python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 1 --batch_size 600 --exp_name 1_worker --project_name bs500_nhid650_drop0.5

#2_workers
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 2 --batch_size 300 --exp_name 2_workers --project_name bs500_nhid650_drop0.5

#4_workers
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 4 --batch_size 150 --exp_name 4_workers --project_name bs500_nhid650_drop0.5

#8_workers
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 8 --batch_size 75 --exp_name 8_workers --project_name bs500_nhid650_drop0.5

#6_workers
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 6 --batch_size 100 --exp_name 6_workers --project_name bs500_nhid650_drop0.5


#1_worker
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 1 --batch_size 600 --exp_name 1_worker --project_name bs500_nhid650_drop0.5_log1

#2_workers
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 2 --batch_size 300 --exp_name 2_workers --project_name bs500_nhid650_drop0.5_log1

#4_workers
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 4 --batch_size 150 --exp_name 4_workers --project_name bs500_nhid650_drop0.5_log1

#8_workers
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 8 --batch_size 75 --exp_name 8_workers --project_name bs500_nhid650_drop0.5_log1

#6_workers
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 6 --batch_size 100 --exp_name 6_workers --project_name bs500_nhid650_drop0.5_log1




python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 1 --batch_size 128 --compress_ratio 1 --exp_name no_compression --project_name bs128_nhid650_drop0.5_compress_decay0
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 2 --batch_size 64 --compress_ratio 1 --exp_name no_compression_2workers --project_name bs128_nhid650_drop0.5_compress_decay0
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 4 --batch_size 32 --compress_ratio 1 --exp_name no_compression_4workers --project_name bs128_nhid650_drop0.5_compress_decay0
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 8 --batch_size 16 --compress_ratio 1 --exp_name no_compression_8workers --project_name bs128_nhid650_drop0.5_compress_decay0

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 1 --batch_size 128 --compress_ratio 0.1 --exp_name TopK_0.1_ResMem --project_name bs128_nhid650_drop0.5_compress_decay0
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 1 --batch_size 128 --compress_ratio 0.01 --exp_name TopK_0.01_ResMem --project_name bs128_nhid650_drop0.5_compress_decay0
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 1 --batch_size 128 --compress_ratio 0.001 --exp_name TopK_0.001_ResMem --project_name bs128_nhid650_drop0.5_compress_decay0
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 1 --batch_size 128 --compress_ratio 0.0001 --exp_name TopK_0.0001_ResMem --project_name bs128_nhid650_drop0.5_compress_decay0

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 1 --batch_size 128 --compress_ratio 0.1 --exp_name TopK_0.1 --project_name bs128_nhid650_drop0.5_compress_decay0
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 1 --batch_size 128 --compress_ratio 0.01 --exp_name TopK_0.01 --project_name bs128_nhid650_drop0.5_compress_decay0
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 1 --batch_size 128 --compress_ratio 0.001 --exp_name TopK_0.001 --project_name bs128_nhid650_drop0.5_compress_decay0

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 1 --batch_size 128 --compress_ratio 0.1 --exp_name RandomK_0.1_ResMem --project_name bs128_nhid650_drop0.5_compress_decay0
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 1 --batch_size 128 --compress_ratio 0.01 --exp_name RandomK_0.01_ResMem --project_name bs128_nhid650_drop0.5_compress_decay0
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 1 --batch_size 128 --compress_ratio 0.001 --exp_name RandomK_0.001_ResMem --project_name bs128_nhid650_drop0.5_compress_decay0

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 1 --batch_size 128 --compress_ratio 0.1 --exp_name RandomK_0.1 --project_name bs128_nhid650_drop0.5_compress_decay0
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 1 --batch_size 128 --compress_ratio 0.01 --exp_name RandomK_0.01 --project_name bs128_nhid650_drop0.5_compress_decay0
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 1 --batch_size 128 --compress_ratio 0.001 --exp_name RandomK_0.001 --project_name bs128_nhid650_drop0.5_compress_decay0

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 2 --batch_size 64 --compress_ratio 0.1 --exp_name TopK_0.1_ResMem_2workers --project_name bs128_nhid650_drop0.5_compress_decay0
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 2 --batch_size 64 --compress_ratio 0.01 --exp_name TopK_0.01_ResMem_2workers --project_name bs128_nhid650_drop0.5_compress_decay0
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 2 --batch_size 64 --compress_ratio 0.001 --exp_name TopK_0.001_ResMem_2workers --project_name bs128_nhid650_drop0.5_compress_decay0

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 8 --batch_size 16 --compress_ratio 0.1 --exp_name TopK_0.1_ResMem_8workers --project_name bs128_nhid650_drop0.5_compress_decay0
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 8 --batch_size 16 --compress_ratio 0.01 --exp_name TopK_0.01_ResMem_8workers --project_name bs128_nhid650_drop0.5_compress_decay0
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 8 --batch_size 16 --compress_ratio 0.001 --exp_name TopK_0.001_ResMem_8workers --project_name bs128_nhid650_drop0.5_compress_decay0

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 4 --batch_size 32 --compress_ratio 0.1 --exp_name TopK_0.1_ResMem_4workers --project_name bs128_nhid650_drop0.5_compress_decay0
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 4 --batch_size 32 --compress_ratio 0.01 --exp_name TopK_0.01_ResMem_4workers --project_name bs128_nhid650_drop0.5_compress_decay0
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 4 --batch_size 32 --compress_ratio 0.001 --exp_name TopK_0.001_ResMem_4workers --project_name bs128_nhid650_drop0.5_compress_decay0
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 100 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 1 --batch_size 128 --compress_ratio 0.0001 --exp_name TopK_0.0001_ResMem_100epochs --project_name bs128_nhid650_drop0.5_compress_decay0

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 1 --batch_size 128 --exp_name OneBit_ResMem --project_name bs128_nhid650_drop0.5_compress_decay0
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 1 --batch_size 128 --exp_name OneBit --project_name bs128_nhid650_drop0.5_compress_decay0

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 100 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 1 --batch_size 128 --compress_ratio 0.001 --compressor topk --memory residual --exp_name TopK_0.001_ResMem --project_name 100epochs_bs128_nhid650_drop0.5_compress_decay0
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 100 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 1 --batch_size 128 --compress_ratio 0.01 --compressor topk --memory residual --exp_name TopK_0.01_ResMem --project_name 100epochs_bs128_nhid650_drop0.5_compress_decay0
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 100 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 1 --batch_size 128 --compress_ratio 0.1 --compressor topk --memory residual --exp_name TopK_0.1_ResMem --project_name 100epochs_bs128_nhid650_drop0.5_compress_decay0
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 100 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 1 --batch_size 128 --compress_ratio 0.00001 --compressor topk --memory residual --exp_name TopK_0.00001_ResMem --project_name 100epochs_bs128_nhid650_drop0.5_compress_decay0
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 100 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 1 --batch_size 128 --compress_ratio 0.00001 --compressor none --memory none --exp_name no_compression --project_name 100epochs_bs128_nhid650_drop0.5_compress_decay0
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 100 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 1 --batch_size 128 --compress_ratio 0.1 --compressor randomk --memory residual --exp_name RandomK_0.1_ResMem --project_name 100epochs_bs128_nhid650_drop0.5_compress_decay0

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 1 --batch_size 128 --compressor none --memory none --exp_name no_compression_1_worker --project_name others_bs128_nhid650_drop0.5_compress_decay0
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 2 --batch_size 64 --compressor none --memory none --exp_name no_compression_2_workers --project_name others_bs128_nhid650_drop0.5_compress_decay0
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 8 --batch_size 16 --compressor none --memory none --exp_name no_compression_8_workers --project_name others_bs128_nhid650_drop0.5_compress_decay0

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 1 --batch_size 128 --compressor onebit --memory none --exp_name OneBit --project_name others_bs128_nhid650_drop0.5_compress_decay0
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 1 --batch_size 128 --compressor onebit --memory residual --exp_name OneBit_ResMem --project_name others_bs128_nhid650_drop0.5_compress_decay0
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 1 --batch_size 128 --compressor terngrad --memory none --exp_name terngrad --project_name others_bs128_nhid650_drop0.5_compress_decay0
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 1 --batch_size 128 --compressor terngrad --memory residual --exp_name terngrad_ResMem --project_name others_bs128_nhid650_drop0.5_compress_decay0
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 100 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 1 --batch_size 128 --compressor terngrad --memory none --exp_name terngrad_100epochs --project_name others_bs128_nhid650_drop0.5_compress_decay0
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 1 --batch_size 128 --compressor terngrad --memory residual --exp_name terngrad_ResMem_AvgGrad --project_name others_bs128_nhid650_drop0.5_compress_decay0
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 1 --batch_size 128 --compressor terngrad --memory none --exp_name terngrad_AvgGrad --project_name others_bs128_nhid650_drop0.5_compress_decay0
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 100 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 1 --batch_size 128 --compressor terngrad --memory none --exp_name terngrad_AvgGrad_100epochs --project_name others_bs128_nhid650_drop0.5_compress_decay0

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 1 --batch_size 128 --compress_ratio 0.1 --compressor dgc --memory none --exp_name dgc_0.1 --project_name others_bs128_nhid650_drop0.5_compress_decay0
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 1 --batch_size 128 --compress_ratio 0.1 --compressor dgc --memory residual --exp_name dgc_0.1_ResMem --project_name others_bs128_nhid650_drop0.5_compress_decay0
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 1 --batch_size 128 --compress_ratio 0.01 --compressor dgc --memory none --exp_name dgc_0.01 --project_name others_bs128_nhid650_drop0.5_compress_decay0
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 1 --batch_size 128 --compress_ratio 0.01 --compressor dgc --memory residual --exp_name dgc_0.01_ResMem --project_name others_bs128_nhid650_drop0.5_compress_decay0
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 1 --batch_size 128 --compress_ratio 0.001 --compressor dgc --memory none --exp_name dgc_0.001 --project_name others_bs128_nhid650_drop0.5_compress_decay0
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 1 --batch_size 128 --compress_ratio 0.001 --compressor dgc --memory residual --exp_name dgc_0.001_ResMem --project_name others_bs128_nhid650_drop0.5_compress_decay0
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 1 --batch_size 128 --compress_ratio 0.0001 --compressor dgc --memory none --exp_name dgc_0.0001 --project_name others_bs128_nhid650_drop0.5_compress_decay0
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 1 --batch_size 128 --compress_ratio 0.0001 --compressor dgc --memory residual --exp_name dgc_0.0001_ResMem --project_name others_bs128_nhid650_drop0.5_compress_decay0

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 1 --batch_size 128 --compress_ratio 0.001 --compressor topk --memory residual --exp_name TopK_0.001_ResMem_AvgGrad --project_name others_bs128_nhid650_drop0.5_compress_decay0
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 1 --batch_size 128 --compress_ratio 0.01 --compressor topk --memory residual --exp_name TopK_0.01_ResMem_AvgGrad --project_name others_bs128_nhid650_drop0.5_compress_decay0
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 1 --batch_size 128 --compress_ratio 0.01 --compressor dgc --memory residual --exp_name dgc_0.01_ResMem_AvgGrad --project_name others_bs128_nhid650_drop0.5_compress_decay0
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 1 --batch_size 128 --compress_ratio 0.001 --compressor dgc --memory residual --exp_name dgc_0.001_ResMem_AvgGrad --project_name others_bs128_nhid650_drop0.5_compress_decay0

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 1 --batch_size 128 --compressor onebit --memory none --exp_name OneBit_AvgGrad --project_name others_bs128_nhid650_drop0.5_compress_decay0
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --epochs 50 --eval_batch_size 10 --test_batch_size 10 --bptt 35 --dropout 0.5 --seed 1111 --log_interval 10 --clip 0.25 --use_gpu true --emb_size 650 --num_hid 650 --num_workers 1 --batch_size 128 --compressor onebit --memory residual --exp_name OneBit_ResMem --project_name others_bs128_nhid650_drop0.5_compress_decay0
