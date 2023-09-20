# SLRZoo: Sign Language Recognition  with deep learning methods


[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)


## Usage


### Requirements

Create a new virtual environment  

```
conda create -n slrzooenv
conda activate slrzooenv
```


and install requirements:

```setup
pip install -r requirements.txt
```


## Training

To train the model(s) for Continuous Sign Language Recognition run this command:

```train
python train_cslr.py  
```

To train the model(s) for Isolated Sign Language Recognition run this command:

```train
python train_islr.py  
```


## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --pretrained-cpkt mymodel.pth 
```



### Download
You can run [download.sh](download.sh) which automatically downloads datasets (except CSL-Daily, whose downloading needs an agreement submission), pretrained models, keypoints and place them under corresponding locations. Or you can download these files separately as follows.

**Datasets**

Download datasets from their websites and place them under the corresponding directories in data/
* [Phoenix-2014](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/)
* [Phoenix-2014T](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/)
* [CSL-Daily](http://home.ustc.edu.cn/~zhouh156/dataset/csl-daily/)

Then run [preprocess/preprocess_video.sh](preprocess/preprocess_video.sh) to extract the downloaded videos. 


### Related work

1. Camgoz, N. C., Hadfield, S., Koller, O., & Bowden, R. (2017, October). Subunets: End-to-end hand shape and continuous sign language recognition. In 2017 IEEE International Conference on Computer Vision (ICCV) (pp. 3075-3084). IEEE.

1. Cui, Runpeng, Hu Liu, and Changshui Zhang. "A deep neural framework for continuous sign language recognition by iterative training." IEEE Transactions on Multimedia 21.7 (2019): 1880-1891.

1. H. Zhou, W. Zhou and H. Li, "Dynamic Pseudo Label Decoding for Continuous Sign Language Recognition," 2019 IEEE International Conference on Multimedia and Expo (ICME), Shanghai, China, 2019, pp. 1282-1287.

1. Pu, Junfu, Wengang Zhou, and Houqiang Li. "Iterative alignment network for continuous sign language recognition." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019.

1. Yang, Zhaoyang, et al. "SF-Net: Structured Feature Network for Continuous Sign Language Recognition." arXiv preprint arXiv:1908.01341 (2019).





[contributors-shield]: https://img.shields.io/github/contributors/iliasprc/slrzoo.svg?style=flat-square
[contributors-url]: https://github.com/iliasprc/slrzoo/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/iliasprc/slrzoo.svg?style=flat-square
[forks-url]: https://github.com/iliasprc/slrzoo/network/members

[stars-shield]: https://img.shields.io/github/stars/iliasprc/slrzoo.svg?style=flat-square
[stars-url]: https://github.com/iliasprc/slrzoo/stargazers

[issues-shield]: https://img.shields.io/github/issues/iliasprc/slrzoo.svg?style=flat-square
[issues-url]: https://github.com/iliasprc/slrzoo/issues



