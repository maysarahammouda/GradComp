# Gradient Compression for Distributed Deep Learning

#### This is an open-source framework that contains the whole code required for compressing language models using PyTorch. The framework is divided into several files to make integrating any new parts very easy. 

___

### Here is a diagram that shows how the code files in the framework are connected:
![framework](https://i.ibb.co/TTBNh0N/framework.png)
___

## Installation
You need to have Python 3 installed.

Create a local environment and install the requirements:
```bash
python3 -m venv .env
source .env/bin/activate
pip install -r requirements.txt
pip install -e .
```
___


### The *datasets* folder contains the Penn Treebank (PTB) corpus. 

___

### The *framework* folder contains the following files:
- **run.sh**: contains the command line code required to run the whole framework. (example below)
- **main.py**: contains the main parts of the code and connects all the files together.
- **model.py**: contains the LSTM model. (can be used to add any other models in the future)
- **model_eval.py**: contains the training, evaluation, and testing code. It also contains the helper code to print the results.
- **optimizer.py**: contains the modified optimizer. (this is currently abandoned and combined with the model_eval code)
- **batch_generation.py**: contains the code for creating the batches from the input data.
- **data_loader.py**: contains the code for reading the input files, tokenizing them, and creating the dictionary. 
- **utils.py**: contains some helper functions that can be used with any other code in the framework.

___

### The *compressor* folder contains the code for the following compressors:
Sparsification | Quantization | Hybrid
------------ | ------------- | -------------
Top-k | EF-SignSGD | Top-k & EF-SignSGD
AdaComp | TernGrad | Top-k & TernGrad
Variance-Based compression | . | AdaComp & EF-SignSGD
. | . | AdaComp & TernGrad

___

### The *memory* folder contains the residual memory and the "no memory" case.

___

### Here is an example on how to run the code:
```python
python main.py --data ../datasets/ptb --num_layers 2 --init_lr 20.0 --bptt 43 --dropout 0.7003 --lr_decay 0.0 --epochs 70 --eval_batch_size 10 --test_batch_size 10 --seed 1111 --log_interval 1 --clip 0.25 --use_gpu true --emb_size 700 --num_hid 700 --num_workers 1 --batch_size 128 --compressor topk --compress_ratio 0.001 --memory residual --exp_name Test_Experiment --project_name Test_Project
```
The details of all of these command line arguments are available in 'main.py'.

___
