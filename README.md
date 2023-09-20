# SLRGAN: Sign Language Recognition with Generative Adversarial Network

This repository is the official implementation of SLRGAN: Sign Language Recognition withGenerative Adversarial Network

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

> 📋Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## Training

To train the model(s) for Continuous Sign Language Recognition run this command:

```train
python train_cslr.py  
```

To train the model(s) for Isolated Sign Language Recognition run this command:

```train
python train_islr.py  
```


> 📋 TODO : Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --pretrained-cpkt mymodel.pth 
```

 
## Usage
### Prerequisites
Create an environment and install dependencies.
```
conda env create -f environment.yml
conda activate slt
```
### Download
You can run [download.sh](download.sh) which automatically downloads datasets (except CSL-Daily, whose downloading needs an agreement submission), pretrained models, keypoints and place them under corresponding locations. Or you can download these files separately as follows.

**Datasets**

Download datasets from their websites and place them under the corresponding directories in data/
* [Phoenix-2014](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/)
* [Phoenix-2014T](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/)
* [CSL-Daily](http://home.ustc.edu.cn/~zhouh156/dataset/csl-daily/)

Then run [preprocess/preprocess_video.sh](preprocess/preprocess_video.sh) to extract the downloaded videos. 

## Reference

If you use this work please cite our work :D :





