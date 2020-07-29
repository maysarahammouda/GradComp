# Gradient Compression for Distributed Deep Learning

#### This is an open-source framework that contains the whole code required for compressing language models. The framework is divided into several codes to make integrating any new parts very easy. 

### Here is a diagram that shows how the codes in the framework are connected:
![framework](https://i.ibb.co/1vprhFY/frameworks.jpg)

### The *datasets* folder contains the Penn Treebank (PTB) corpus. 

### The *framework* folder contains the following files:
- **run.py**: contains the command line code required to run the whole framework. (example below)
- **main.py**: contains the main parts of the code and connects all the files together.
- **model.py**: contains the LSTM model. (can be used to add any other models in the future)
- **model_eval.py**: contains the training, evaluation, and testing code. It also contains the helper code to print the results.
- **optimizer.py**: contains the modified optimizer. (this is currently abandoned and combined with the model_eval code)
- **batch_generation.py**: contains the code for creating the batches from the input data.
- **data_loader.py**: contains the code for reading the input files, tokenizing them, and creating the dictionary. 
- **utils.py**: contains some helper functions that can be used with any other code in the framework.
### The *compressor* folder contains the code for the following compressors:

Sparsification | Quantization | Hybrid
------------ | ------------- | -------------
Top-k | EF-SignSGD | Top-k & EF-SignSGD
AdaComp | TernGrad | Top-k & TernGrad
Variance-Based compression | . | AdaComp & EF-SignSGD
. | . | AdaComp & TernGrad
