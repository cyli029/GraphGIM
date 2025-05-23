<h1 align="center">  GraphGIM: Rethinking Molecular Graph Contrastive Learning via Geometry Image Modeling  </h1>
Official PyTorch-based implementation in the paper GraphGIM: Rethinking Molecular Graph Contrastive Learning via Geometry Image Modeling, which contains a simple contrastive learning framework called GraphGIM and two variants (GraphGIM-M and GraphGIM-P).
<div>
<a href="https://github.com/cyli029/GraphGIM/blob/main/LICENSE">
    <img alt="GitHub" src="https://img.shields.io/github/license/cyli029/GraphGIM?style=flat-square">
</a><img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/cyli029/GraphGIM?style=flat-square">
</div>


## 🆕 News

- \[**August 2024**\] We propose [GraphGIM](https://github.com/cyli029/GraphGIM), a Molecular Graph Contrastive Learning Framework.
- \[**May 15 2025**\] Congratulations on our work being accepted by BMC biology.

## Abstract
**Background:** Learning molecular representations is crucial for accurate drug discovery. Using graphs to represent molecules is a popular solution, and many researchers have used contrastive learning to improve the generalization of molecular graph representations. 

**Results:** In this work, we revisit existing graph-based contrastive methods and find that these methods have limited diversity in the constructed sample pairs, resulting in insufficient performance gains. To alleviate the above challenge, we propose a novel molecular graph contrastive learning method via geometry image modeling, called GraphGIM, which enhances the diversity between sample pairs. GraphGIM is pre-trained on 2 million 2D graphs and multi-view 3D geometry images through contrastive learning. Furthermore, we find that as the convolutional layers process the image becomes deeper, the information of feature maps gradually changes from global molecular-level information (molecular scaffolds) to local atomic-level information (molecular atoms and functional groups), which provides chemical information at different scales. Therefore, we propose two variants of GraphGIM, called GraphGIM-M and GraphGIM-P, which fuse feature maps of different scales in the image using a weighted strategy and a prompt-based strategy, respectively.

**Conclusions:**
Extensive experiments show that GraphGIM and its two variants outperform state-of-the-art graph contrastive learning methods on eight molecular property prediction benchmarks from MoleculeNet and achieve competitive results with state-of-the-art methods. 


![framework](./assets/framework.png)

## Environments

#### 1. GPU environment
CUDA 11.6
Ubuntu 20.04
Miniconda conda3
Python 3.8

#### 2. create conda environment
```bash
# create conda env
conda create -n GraphGIM python=3.8.10
source activate GraphGIM
# install environment
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116 -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install timm==1.0.3
pip install tensorboard
pip install scikit-learn
pip install pandas
pip install numpy
pip install torch-cluster 1.6.0+pt113cu116 torch-scatter torch-sparse torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.13.1%2Bcu116.html
pip install torch-geometric==2.5.3
pip install dgl-1.0.0+cu116
pip install ogb
pip install tqdm
```
#### 3. Pre-training the model by contrastive learning (GraphGIM)
You can download [pretraining data](https://pan.baidu.com/s/1C8l3_tjd8dFS5oGvXqHcxQ?pwd=pdq4) and push it into the folder `dataset/pre-training/`.
It is worth noting that you need to pre-process the dataset to ensure data availability.
**Tip：s** During the dataset processing, if you find the above dataset lacks some images, you can download the extra images from [supplement data](https://pan.quark.cn/s/1489b7d2a226);
#### 4. Downstream tasks from MoleculeNet
You can unzip the file in the folder `dataset/downstream/` and put the files into the folder `dataset/downstream/`
#### 5. Extracting fetures of image3d and graph（w/o aug and aug）
You can download [feature data](https://pan.quark.cn/s/cbe0b82864b2) and use it for your tasks.

🔥The pre-trained teacher model can be accessed in the following table.
| Name                | Download link                                                | Description                                                  |
| ------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Pre-trained teacher | [IEM.pth](https://pan.quark.cn/s/bc69a5d6875a) | You can download the teacher and put it in the directory: `resumes/pretrained-model/`. |

🔥The pre-trained GraphGIM model can be accessed in the following table.
| Name                | Download link                                                | Description                                                  |
| ------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| best_epoch=9_loss=0.04 | [GraphGIM.pth](https://pan.quark.cn/s/a0bf837562a1) | You can download the teacher and put it in the directory: `logs/pretraining/`. |
