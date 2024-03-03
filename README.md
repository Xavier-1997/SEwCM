## Usage

Please refer to the official repo for details of data preparation and hardware configurations.

- Prerequisites: Python3.10, [pytorch=2.1](http://pytorch.org), Numpy, TensorboardX, Scikit_learn

- Clone this repo: `git clone https://github.com/Xavier-1997/SEwCM`


## Training on CUB200 dataset

The training and testing set DO NOT share any common categories. We use recall at K accuracy to evaluate the performance following existing unsupervised embedding learning papers.

#### 1) Dataset Preparation

  Prepare the data. Download the datasets first. Then run codes `./pre_process_bird.py` to download and pre-process each dataset

  You may use 
  ```bash
  python pre_process_cub.py
  ```
  Alternatively, you can use the [code](https://github.com/ColumbiaDVMM/Heated_Up_Softmax_Embedding/tree/master/dataset) to directly  download and pre-process the datasets.

  **Remember to change the dataset path to your own path.**

#### 2) Start Training

```bash
python train_test.py --dataset cub200 --arch inception_v1_ml --lr 0.001 --margin 0.5 --low-dim 128 --batch-size 64 --gpu 0
```

  - `--dataset`: "cub200": CUB200-2011 dataset or, "car196": Cars196 dataset, "ebay": Stanford Online Product dataset.
  
  - `--arch`: backbone network structure (defualt: inception_v1_ml).
  
  - `--lr`: learning rate (initialization: 0.01).

  - `--margin`: Size of margin.
  
  - `--gpu`: gpu used for training (only support single GPU training for inception_v1 backbone)

