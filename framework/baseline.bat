#Different batch sizes - Baseline- *k
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5 --bptt 27 --dropout 0.6429 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 32 --compressor none --memory none --exp_name No_Compression_BS_32_k --project_name batch_effect

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 10 --bptt 27 --dropout 0.6429 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 64 --compressor none --memory none --exp_name No_Compression_BS_64_k --project_name batch_effect

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 20 --bptt 27 --dropout 0.6429 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 128 --compressor none --memory none --exp_name No_Compression_BS_128_k --project_name batch_effect

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 40 --bptt 27 --dropout 0.6429 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 256 --compressor none --memory none --exp_name No_Compression_BS_256_k --project_name batch_effect

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 80 --bptt 27 --dropout 0.6429 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 512 --compressor none --memory none --exp_name No_Compression_BS_512_k --project_name batch_effect

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 56.6 --bptt 27 --dropout 0.6429 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 160 --num_hid 700 --num_workers 1 --batch_size 1024 --compressor none --memory none --exp_name No_Compression_BS_1024_k --project_name batch_effect
