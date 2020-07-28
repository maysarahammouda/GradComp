#No_Compression
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 20.0 --bptt 27 --dropout 0.6429 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 128 --compressor none --memory none --exp_name No_Compression_1_worker --project_name multi_workers_1

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 20.0 --bptt 27 --dropout 0.6429 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 2 --batch_size 64 --compressor none --memory none --exp_name No_Compression_2_workers --project_name multi_workers_1

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 20.0 --bptt 27 --dropout 0.6429 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 4 --batch_size 32 --compressor none --memory none --exp_name No_Compression_4_workers --project_name multi_workers_1

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 20.0 --bptt 27 --dropout 0.6429 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 8 --batch_size 16 --compressor none --memory none --exp_name No_Compression_8_workers --project_name multi_workers_1



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


#TopK_0.001
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 20.0 --bptt 43 --dropout 0.7003 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 128 --compressor topk --compress_ratio 0.001 --memory residual --exp_name TopK_0.001_1_worker --project_name multi_workers_1

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 20.0 --bptt 43 --dropout 0.7003 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 2 --batch_size 64 --compressor topk --compress_ratio 0.001 --memory residual --exp_name TopK_0.001_2_workers --project_name multi_workers_1

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 20.0 --bptt 43 --dropout 0.7003 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 4 --batch_size 32 --compressor topk --compress_ratio 0.001 --memory residual --exp_name TopK_0.001_4_workers --project_name multi_workers_1

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 20.0 --bptt 43 --dropout 0.7003 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 8 --batch_size 16 --compressor topk --compress_ratio 0.001 --memory residual --exp_name TopK_0.001_8_workers --project_name multi_workers_1


#TernGrad
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --bptt 20 --dropout 0.4854 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 128 --compressor terngrad --memory none --clip_const 25.673 --exp_name TernGrad_1_worker --project_name multi_workers_1

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --bptt 20 --dropout 0.4854 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 2 --batch_size 64 --compressor terngrad --memory none --clip_const 25.673 --exp_name TernGrad_2_workers --project_name multi_workers_1

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --bptt 20 --dropout 0.4854 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 4 --batch_size 32 --compressor terngrad --memory none --clip_const 25.673 --exp_name TernGrad_4_workers --project_name multi_workers_1

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --bptt 20 --dropout 0.4854 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 8 --batch_size 16 --compressor terngrad --memory none --clip_const 25.673 --exp_name TernGrad_8_workers --project_name multi_workers_1


#AdaComp
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 7.0 --bptt 15 --dropout 0.4928 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 128 --compressor adacomp --memory residual --comp_const 2.35 --exp_name AdaComp_1_worker --project_name multi_workers_1

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 7.0 --bptt 15 --dropout 0.4928 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 2 --batch_size 64 --compressor adacomp --memory residual --comp_const 2.35 --exp_name AdaComp_2_workers --project_name multi_workers_1

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 7.0 --bptt 15 --dropout 0.4928 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 4 --batch_size 32 --compressor adacomp --memory residual --comp_const 2.35 --exp_name AdaComp_4_workers --project_name multi_workers_1

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 7.0 --bptt 15 --dropout 0.4928 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 8 --batch_size 16 --compressor adacomp --memory residual --comp_const 2.35 --exp_name AdaComp_8_workers --project_name multi_workers_1

###########################################################################################################################################

#Adacomp_different_seeds
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 7.0 --bptt 15 --dropout 0.4928 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 2222 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 128 --compressor adacomp --memory residual --comp_const 2.35 --exp_name adaComp_2222 --project_name random_search

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 7.0 --bptt 15 --dropout 0.4928 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 3333 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 128 --compressor adacomp --memory residual --comp_const 2.35 --exp_name adaComp_3333 --project_name random_search


###########################################################################################################################################

#Different batch sizes - Baseline
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 10.0 --bptt 27 --dropout 0.6429 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 32 --compressor none --memory none --exp_name No_Compression_BS_32 --project_name batch_effect

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 14.1 --bptt 27 --dropout 0.6429 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 64 --compressor none --memory none --exp_name No_Compression_BS_64 --project_name batch_effect

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 20.0 --bptt 27 --dropout 0.6429 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 128 --compressor none --memory none --exp_name No_Compression_BS_128 --project_name batch_effect

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 28.3 --bptt 27 --dropout 0.6429 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 256 --compressor none --memory none --exp_name No_Compression_BS_256 --project_name batch_effect

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 40.0 --bptt 27 --dropout 0.6429 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 512 --compressor none --memory none --exp_name No_Compression_BS_512 --project_name batch_effect

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 56.6 --bptt 27 --dropout 0.6429 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 1024 --compressor none --memory none --exp_name No_Compression_BS_1024 --project_name batch_effect


#Different batch sizes - EF-SignSGD
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 10.5 --bptt 29 --dropout 0.6904 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 32 --compressor efsignsgd --memory residual --exp_name EF_SignSGD_BS_32 --project_name batch_effect

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 14.8 --bptt 29 --dropout 0.6904 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 64 --compressor efsignsgd --memory residual --exp_name EF_SignSGD_BS_64 --project_name batch_effect

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 21.0 --bptt 29 --dropout 0.6904 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 128 --compressor efsignsgd --memory residual --exp_name EF_SignSGD_BS_128 --project_name batch_effect

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 29.7 --bptt 29 --dropout 0.6904 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 256 --compressor efsignsgd --memory residual --exp_name EF_SignSGD_BS_256 --project_name batch_effect

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 42 --bptt 29 --dropout 0.6904 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 512 --compressor efsignsgd --memory residual --exp_name EF_SignSGD_BS_512 --project_name batch_effect

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 59.4 --bptt 29 --dropout 0.6904 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 1024 --compressor efsignsgd --memory residual --exp_name EF_SignSGD_BS_1024 --project_name batch_effect


#Different batch sizes - TernGrad
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 2.5 --bptt 20 --dropout 0.4854 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 32 --compressor terngrad --memory none --exp_name TernGrad_BS_32 --project_name batch_effect

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 3.5 --bptt 20 --dropout 0.4854 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 64 --compressor terngrad --memory none --exp_name TernGrad_BS_64 --project_name batch_effect

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 5.0 --bptt 20 --dropout 0.4854 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 128 --compressor terngrad --memory none --exp_name TernGrad_BS_128 --project_name batch_effect


########################################################################################################

#Top-K_EF-EF_SignSGD
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 20.0 --bptt 43 --dropout 0.7003 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 128 --compressor efsigntopk --compress_ratio 0.001 --memory residual --exp_name EF_SignSGD_TopK_0.001 --project_name hybrid

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 20.0 --bptt 43 --dropout 0.7003 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 128 --compressor efsigntopk --compress_ratio 0.01 --memory residual --exp_name EF_SignSGD_TopK_0.01 --project_name hybrid

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 20.0 --bptt 43 --dropout 0.7003 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 128 --compressor efsigntopk --compress_ratio 0.1 --memory residual --exp_name EF_SignSGD_TopK_0.1 --project_name hybrid

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 7.0 --bptt 15 --dropout 0.4928 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 128 --compressor efsignadacomp --memory residual --comp_const 2.35 --exp_name EF-SignSGD_AdaComp --project_name hybrid

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 21.0 --bptt 29 --dropout 0.6904 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 128 --compressor efsigntopk --compress_ratio 0.001 --memory residual --exp_name EF_SignSGD_TopK_0.001_2 --project_name hybrid

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 21.0 --bptt 29 --dropout 0.6904 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 128 --compressor efsigntopk --compress_ratio 0.01 --memory residual --exp_name EF_SignSGD_TopK_0.01_2 --project_name hybrid

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 21.0 --bptt 29 --dropout 0.6904 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 128 --compressor efsigntopk --compress_ratio 0.1 --memory residual --exp_name EF_SignSGD_TopK_0.1_2 --project_name hybrid

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 21.0 --bptt 29 --dropout 0.6904 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 128 --compressor efsignadacomp --memory residual --comp_const 2.35 --exp_name EF-SignSGD_AdaComp_2 --project_name hybrid

python main.py --data ../datasets/ptb --num_layers 2 --init_lr 8 --bptt 11 --dropout 0.5729 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 128 --compressor efsignadacomp --memory residual --comp_const 2.35 --exp_name EF-SignSGD_AdaComp_best --project_name hybrid
