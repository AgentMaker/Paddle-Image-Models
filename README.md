# Paddle-Image-Models
![GitHub forks](https://img.shields.io/github/forks/AgentMaker/Paddle-Image-Models)
![GitHub Repo stars](https://img.shields.io/github/stars/AgentMaker/Paddle-Image-Models)
![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/AgentMaker/Paddle-Image-Models?include_prereleases)
![GitHub](https://img.shields.io/github/license/AgentMaker/Paddle-Image-Models)  

English | [简体中文](README_CN.md)

A PaddlePaddle version image model zoo.

## Install Package
* Install by pip：

    ```shell
    $ pip install ppim==1.0.4 -i https://pypi.python.org/pypi 
    ```
* Install by wheel package：[【Releases Packages】](https://github.com/AgentMaker/Paddle-Image-Models/releases)

## Usage
* Quick Start

    ```python
    import paddle
    from ppim import rednet26

    # Load the model
    model, val_transforms = rednet26(pretrained=True)

    # Model summary 
    paddle.summary(model, input_size=(1, 3, 224, 224))

    # Random a input
    x = paddle.randn(shape=(1, 3, 224, 224))

    # Model forword
    out = model(x)
    ```
* Finetune
    
    ```python
    import paddle
    import paddle.nn as nn
    import paddle.vision.transforms as T
    from paddle.vision import Cifar100

    from ppim import rexnet_100

    # Load the model
    model, val_transforms = rexnet_100(pretrained=True)

    # Use the PaddleHapi Model
    model = paddle.Model(model)

    # Set the optimizer
    opt = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())

    # Set the loss function
    loss = nn.CrossEntropyLoss()

    # Set the evaluate metric
    metric = paddle.metric.Accuracy(topk=(1, 5))

    # Prepare the model 
    model.prepare(optimizer=opt, loss=loss, metrics=metric)

    # Set the data preprocess
    train_transforms = T.Compose([
        T.Resize(256, interpolation='bicubic'),
        T.RandomCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the Cifar100 dataset
    train_dataset = Cifar100(mode='train', transform=train_transforms, backend='pil')
    val_dataset = Cifar100(mode='test',  transform=val_transforms, backend='pil')

    # Finetune the model 
    model.fit(
        train_data=train_dataset, 
        eval_data=val_dataset, 
        batch_size=256, 
        epochs=2, 
        eval_freq=1, 
        log_freq=1, 
        save_dir='save_models', 
        save_freq=1, 
        verbose=1, 
        drop_last=False, 
        shuffle=True,
        num_workers=0
    )
    ```

## Model Zoo
* ReXNet
    * Paper：[ReXNet: Diminishing Representational Bottleneck on Convolutional Neural Network](https://arxiv.org/abs/2007.00992)
    * Origin Repo：[clovaai/rexnet](https://github.com/clovaai/rexnet)
    * Evaluate Transforms：
        ```python
        # backend: pil
        # input_size: 224x224
        transforms = T.Compose([
            T.Resize(256, interpolation='bicubic'),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        ```
    * Model Details：

        |         Model         | Params(M) | FLOPs(G) | Top-1 (%) | Top-5 (%) |
        |:---------------------:|:---------:|:--------:|:---------:|:---------:|
        | ReXNet-1.0             |  4.8 | 0.40 | 77.9 | 93.9 |
        | ReXNet-1.3             |  7.6 | 0.66 | 79.5 | 94.7 |
        | ReXNet-1.5             |  7.6 | 0.66 | 80.3 | 95.2 |
        | ReXNet-2.0             |  16  | 1.5  | 81.6 | 95.7 |
        | ReXNet-3.0             |  34  | 3.4  | 82.8 | 96.2 |

* RedNet
    * Paper：[Involution: Inverting the Inherence of Convolution for Visual Recognition](https://arxiv.org/abs/2103.06255)
    * Origin Repo：[d-li14/involution](https://github.com/d-li14/involution)
    * Evaluate Transforms：
        ```python
        # backend: cv2
        # input_size: 224x224
        transforms = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.Normalize(
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True,
                data_format='HWC'
            ),
            T.ToTensor(),
        ])
        ```
    * Model Details：

        |         Model         | Params(M) | FLOPs(G) | Top-1 (%) | Top-5 (%) |
        |:---------------------:|:---------:|:--------:|:---------:|:---------:|
        | RedNet-26             |  9.23 | 1.73 | 75.96 | 93.19 |
        | RedNet-38             | 12.39 | 2.22 | 77.48 | 93.57 |
        | RedNet-50             | 15.54 | 2.71 | 78.35 | 94.13 |
        | RedNet-101            | 25.65 | 4.74 | 78.92 | 94.35 |
        | RedNet-152            | 33.99 | 6.79 | 79.12 | 94.38 |

* RepVGG
    * Paper：[RepVGG: Making VGG-style ConvNets Great Again](https://arxiv.org/abs/2101.03697)
    * Origin Repo：[DingXiaoH/RepVGG](https://github.com/DingXiaoH/RepVGG)
    * Evaluate Transforms：
        ```python
        # backend: pil
        # input_size: 224x224
        transforms = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        ```
    * Model Details：

        |         Model         | Params(M) | FLOPs(G) | Top-1 (%) | Top-5 (%) |
        |:---------------------:|:---------:|:--------:|:---------:|:---------:|
        | RepVGG-A0             |  8.30  | 1.4  | 72.41 |       |
        | RepVGG-A1             | 12.78  | 2.4  | 74.46 |       |
        | RepVGG-A2             | 25.49  | 5.1  | 76.48 |       |
        | RepVGG-B0             | 14.33  | 3.1  | 75.14 |       |
        | RepVGG-B1             | 51.82  | 11.8 | 78.37 |       |
        | RepVGG-B2             | 80.31  | 18.4 | 78.78 |       |
        | RepVGG-B3             | 110.96 | 26.2 | 80.52 |       |
        | RepVGG-B1g2           | 41.36  | 8.8  | 77.78 |       |
        | RepVGG-B1g4           | 36.12  | 7.3  | 77.58 |       |
        | RepVGG-B2g4           | 55.77  | 11.3 | 79.38 |       |
        | RepVGG-B3g4           | 75.62  | 16.1 | 80.21 |       |
    
* DeiT
    * Paper：[Training data-efficient image transformers & distillation through attention
](https://arxiv.org/abs/2012.12877)
    * Origin Repo：[facebookresearch/deit](https://github.com/facebookresearch/deit)
    * Evaluate Transforms：
        ```python
        # backend: pil
        # input_size: 224x224
        transforms = T.Compose([
            T.Resize(248, interpolation='bicubic'),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # backend: pil
        # input_size: 384x384
        transforms = T.Compose([
            T.Resize(384, interpolation='bicubic'),
            T.CenterCrop(384),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        ```
    * Model Details：

        |         Model         | Params(M) | FLOPs(G) | Top-1 (%) | Top-5 (%) |
        |:---------------------:|:---------:|:--------:|:---------:|:---------:|
        | DeiT-tiny               |  5  | | 72.2 |  91.1     |
        | DeiT-small              | 22  | | 79.9 |  95.0     |
        | DeiT-base               | 86  | | 81.8 |  95.6     |
        | DeiT-tiny distilled     | 6   | | 74.5 |  91.9     |
        | DeiT-small distilled    | 22  | | 81.2 |  95.4     |
        | DeiT-base distilled     | 87  | | 83.4 |  96.5     |
        | DeiT-base 384           | 87  | | 82.9 |  96.2     |
        | DeiT-base distilled 384 | 88  | | 85.2 |  97.2     |

## Citation
```
@article{han2020rexnet,
    title = {{ReXNet}: Diminishing Representational Bottleneck on Convolutional Neural Network},
    author = {Han, Dongyoon and Yun, Sangdoo and Heo, Byeongho and Yoo, YoungJoon},
    journal = {arXiv preprint arXiv:2007.00992},
    year = {2020},
}
```
```
@InProceedings{Li_2021_CVPR,
    title = {Involution: Inverting the Inherence of Convolution for Visual Recognition},
    author = {Li, Duo and Hu, Jie and Wang, Changhu and Li, Xiangtai and She, Qi and Zhu, Lei and Zhang, Tong and Chen, Qifeng},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2021}
}
```
```
@article{ding2021repvgg,
    title={RepVGG: Making VGG-style ConvNets Great Again},
    author={Ding, Xiaohan and Zhang, Xiangyu and Ma, Ningning and Han, Jungong and Ding, Guiguang and Sun, Jian},
    journal={arXiv preprint arXiv:2101.03697},
    year={2021}
}
```
```
@article{touvron2020deit,
    title = {Training data-efficient image transformers & distillation through attention},
    author = {Hugo Touvron and Matthieu Cord and Matthijs Douze and Francisco Massa and Alexandre Sablayrolles and Herv'e J'egou},
    journal = {arXiv preprint arXiv:2012.12877},
    year = {2020}
}

```
