# RS-CCL

Official PyTorch implementation of paper “Federated Coupled Contrastive Learning on Remote Sensing Image Classification”.

## Abstract

Classification for Remote Sensing (RS) images plays a crucial role in various fields such as disaster assessment and urban planning. Nevertheless, due to resource limitations and privacy issues, directly transferring data to the ground for high-performance centralized training is often impractical. Federated Learning (FL) has emerged as a revolutionary distributed learning paradigm, especially prominent in privacy-sensitive scenarios. However, since the RS data is distributed across numerous participants, FL suffers from an inevitable challenge of data heterogeneity for RS image classification. As the local data is more unevenly distributed, this challenge becomes increasingly pronounced, leading to inconsistent local update directions. In this paper, we propose RS-CCL, a novel Federated Coupled Contrastive Learning rule on RS image classification, which expands the feature space, thereby allowing models to better learn the intrinsic structure of the complex RS data effectively. Our RS-CCL mathematically formulates cross-entropy and feature similarity into a unified framework. It is then solved by coupled gradient computation in a consistent optimization direction, overcoming dimensional collapse. Within this framework, we further introduce a dynamic fusion mask matrix, which inhibits the model from excessively drawing on inter-class similarity. Empirically, we show that our RS-CCL is in line with robustness theory that a larger feature space brings more robust performance. Extensive experiments on several benchmarks validate the effectiveness and generalization ability of our method. For example, with a high data heterogeneity of $\alpha=0.05$, our RS-CCL outperforms other baselines by up to 8.5\% on UC Merced dataset with MobileViT-S.

## Environment
We conduct our experiments on 3090 GPUs in an environment configured as follows:
- Python version: 3.11
- PyTorch version: 2.2.2+cu118

Other requirements are shown in `requirements.txt`


## Configurations
* See `config.yaml`

## Run
Run the following command, you can reproduce RS-CCL on UC Merced and NWPU datasets. 
* `python main.py`

The values of `dataset_name` and `num_classes` may need to be adjusted in the `config.yaml` file according to different datasets. Other settings related to federated learning as well as the hyperparameters of RS-CCL can also be adjusted here.

## Results
For brevity, we present only the results of RS-CCL and some comparison methods.

| Model    | Method                                   | α = 0.05            | α = 0.1             | α = 0.5             | IID                 |
|----------|------------------------------------------|---------------------|---------------------|---------------------|---------------------|
| ResNet18 | FedAvg [1]                               | 79.75±2.2           | 80.16±2.0           | 81.50±1.8           | 82.74±1.0           |
|          | FedProx [2]                              | 80.22±1.5           | 80.53±2.3           | 81.22±1.4           | 82.20±0.8           |
|          | **RS-CCL**                               | **87.17±1.5**       | **88.07±1.5**       | **90.56±0.8**       | **90.82±0.7**       |
| VGG16    | FedAvg [1]                               | 78.90±2.3           | 79.40±2.1           | 80.80±1.9           | 82.10±1.1           |
|          | FedProx [2]                              | 79.40±1.6           | 79.80±2.4           | 80.50±1.5           | 81.60±0.9           |
|          | **RS-CCL**                               | **86.00±1.8**       | **86.90±1.6**       | **89.30±0.9**       | **89.50±1.0**       |
| MobileViT-S| FedAvg [1]                               | 80.30±2.4           | 80.70±2.2           | 82.00±2.0           | 83.10±1.2           |
|          | FedProx [2]                              | 80.80±1.7           | 81.20±2.5           | 81.90±1.6           | 82.60±1.0           |
|          | **RS-CCL**                               | **87.50±2.1**       | **88.40±1.6**       | **90.90±1.3**       | **91.10±0.8**       |

*Table: Comparison on UC Merced Dataset.*

**References:**  
[1] McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data", AISTATS 2017  
[2] Li et al., "Federated Optimization in Heterogeneous Networks", MLSys 2020  
