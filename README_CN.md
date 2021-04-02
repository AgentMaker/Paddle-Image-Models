# Paddle-Image-Models
![GitHub forks](https://img.shields.io/github/forks/AgentMaker/Paddle-Image-Models)
![GitHub Repo stars](https://img.shields.io/github/stars/AgentMaker/Paddle-Image-Models)
![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/AgentMaker/Paddle-Image-Models?include_prereleases)
![GitHub](https://img.shields.io/github/license/AgentMaker/Paddle-Image-Models)  

[English](README.md) | 简体中文

一个基于飞桨框架实现的图像预训练模型库。

## 安装
* 通过 pip 进行安装：

    ```shell
    $ pip install ppim==1.0.5 -i https://pypi.python.org/pypi 
    ```

* 通过 whl 包进行安装：[【Releases Packages】](https://github.com/AgentMaker/Paddle-Image-Models/releases)

## 使用方法
* 快速使用

    ```python
    import paddle
    from ppim import rednet_26

    # 加载模型
    model, val_transforms = rednet_26(pretrained=True)

    # 模型结构总览 
    paddle.summary(model, input_size=(1, 3, 224, 224))

    # 准备一个随机的输入
    x = paddle.randn(shape=(1, 3, 224, 224))

    # 模型前向计算
    out = model(x)
    ```

* 模型微调
    
    ```python
    import paddle
    import paddle.nn as nn
    import paddle.vision.transforms as T
    from paddle.vision import Cifar100

    from ppim import rexnet_1_0

    # 加载模型
    model, val_transforms = rexnet_1_0(pretrained=True)

    # 使用飞桨高层 API Model
    model = paddle.Model(model)

    # 配置优化器
    opt = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())

    # 配置损失函数
    loss = nn.CrossEntropyLoss()

    # 配置评估指标
    metric = paddle.metric.Accuracy(topk=(1, 5))

    # 模型准备
    model.prepare(optimizer=opt, loss=loss, metrics=metric)

    # 配置训练集数据处理
    train_transforms = T.Compose([
        T.Resize(256, interpolation='bicubic'),
        T.RandomCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载 Cifar100 数据集
    train_dataset = Cifar100(mode='train', transform=train_transforms, backend='pil')
    val_dataset = Cifar100(mode='test',  transform=val_transforms, backend='pil')

    # 模型微调
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

## 模型库
### ReXNet
* 论文：[ReXNet: Diminishing Representational Bottleneck on Convolutional Neural Network](https://arxiv.org/abs/2007.00992)
* 官方项目：[clovaai/rexnet](https://github.com/clovaai/rexnet)
* 验证集数据处理：

    ```python
    # 图像后端：pil
    # 输入图像大小：224x224
    transforms = T.Compose([
        T.Resize(256, interpolation='bicubic'),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    ```

* 模型细节：

    |         Model         |     Model Name        | Params(M) | FLOPs(G) | Top-1 (%) | Top-5 (%) |
    |:---------------------:|:---------------------:|:---------:|:--------:|:---------:|:---------:|
    | ReXNet-1.0            | rexnet_1_0            |  4.8      | 0.40     | 77.86     | 93.87     |
    | ReXNet-1.3            | rexnet_1_3            |  7.6      | 0.66     | 79.50     | 94.68     |
    | ReXNet-1.5            | rexnet_1_5            |  7.6      | 0.66     | 80.32     | 95.17     |
    | ReXNet-2.0            | rexnet_2_0            |  16       | 1.5      | 81.64     | 95.66     |
    | ReXNet-3.0            | rexnet_3_0            |  34       | 3.4      | 82.45     | 96.26     |

### RedNet
* 论文：[Involution: Inverting the Inherence of Convolution for Visual Recognition](https://arxiv.org/abs/2103.06255)
* 官方项目：[d-li14/involution](https://github.com/d-li14/involution)
* 验证集数据处理：

    ```python
    # 图像后端：cv2
    # 输入图像大小：224x224
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

* 模型细节：

    |         Model         |     Model Name        | Params(M) | FLOPs(G) | Top-1 (%) | Top-5 (%) |
    |:---------------------:|:---------------------:|:---------:|:--------:|:---------:|:---------:|
    | RedNet-26             |     rednet_26         |  9.23     | 1.73     | 75.96     | 93.19     |
    | RedNet-38             |     rednet_38         | 12.39     | 2.22     | 77.48     | 93.57     |
    | RedNet-50             |     rednet_50         | 15.54     | 2.71     | 78.35     | 94.18     |
    | RedNet-101            |     rednet_101        | 25.65     | 4.74     | 78.92     | 94.35     |
    | RedNet-152            |     rednet_152        | 33.99     | 6.79     | 79.12     | 94.38     |

### RepVGG
* 论文：[RepVGG: Making VGG-style ConvNets Great Again](https://arxiv.org/abs/2101.03697)
* 官方项目：[DingXiaoH/RepVGG](https://github.com/DingXiaoH/RepVGG)
* 验证集数据处理：

    ```python
    # 图像后端：pil
    # 输入图像大小：224x224
    transforms = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    ```

* 模型细节：

    |         Model         |     Model Name        | Params(M) | FLOPs(G) | Top-1 (%) | Top-5 (%) |
    |:---------------------:|:---------------------:|:---------:|:--------:|:---------:|:---------:|
    | RepVGG-A0             | repvgg_a0             |  8.30     | 1.4      | 72.41     |           |
    | RepVGG-A1             | repvgg_a1             | 12.78     | 2.4      | 74.46     |           |
    | RepVGG-A2             | repvgg_a2             | 25.49     | 5.1      | 76.48     |           |
    | RepVGG-B0             | repvgg_b0             | 14.33     | 3.1      | 75.14     |           |
    | RepVGG-B1             | repvgg_b1             | 51.82     | 11.8     | 78.37     |           |
    | RepVGG-B2             | repvgg_b2             | 80.31     | 18.4     | 78.78     |           |
    | RepVGG-B3             | repvgg_b3             | 110.96    | 26.2     | 80.52     |           |
    | RepVGG-B1g2           | repvgg_b1g2           | 41.36     | 8.8      | 77.78     |           |
    | RepVGG-B1g4           | repvgg_b1g4           | 36.12     | 7.3      | 77.58     |           |
    | RepVGG-B2g4           | repvgg_b2g4           | 55.77     | 11.3     | 79.38     |           |
    | RepVGG-B3g4           | repvgg_b3g4           | 75.62     | 16.1     | 80.21     |           |

### PiT
* 论文：[Rethinking Spatial Dimensions of Vision Transformers](https://arxiv.org/abs/2103.16302)
* 官方项目：[naver-ai/pit](https://github.com/naver-ai/pit)
* 验证集数据处理：

    ```python
    # 图像后端：pil
    # 输入图像大小：224x224
    transforms = T.Compose([
        T.Resize(248, interpolation='bicubic'),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    ```

* 模型细节：

    |         Model         |         Model         | Params(M) | FLOPs(G) | Top-1 (%) | Top-5 (%) |
    |:---------------------:|:---------------------:|:---------:|:--------:|:---------:|:---------:|
    | PiT-Ti                | pit_ti                | 4.9       | 0.71     | 73.0      |           |
    | PiT-XS                | pit_xs                | 10.6      | 1.4      | 78.1      |           |
    | PiT-S                 | pit_s                 | 23.5      | 2.9      | 80.9      |           |
    | PiT-B                 | pit_b                 | 73.8      | 12.5     | 82.0      |           |
    | PiT-Ti distilled      | pit_ti_distilled      | 4.9       | 0.71     | 74.6      |           |
    | PiT-XS distilled      | pit_xs_distilled      | 10.6      | 1.4      | 79.1      |           |
    | PiT-S distilled       | pit_s_distilled       | 23.5      | 2.9      | 81.9      |           |
    | PiT-B distilled       | pit_b_distilled       | 73.8      | 12.5     | 84.0      |           |

### DeiT
* 论文：[Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877)
* 官方项目：[facebookresearch/deit](https://github.com/facebookresearch/deit)
* 验证集数据处理：

    ```python
    # 图像后端：pil
    # 输入图像大小：224x224
    transforms = T.Compose([
        T.Resize(248, interpolation='bicubic'),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 图像后端：pil
    # 输入图像大小：384x384
    transforms = T.Compose([
        T.Resize(384, interpolation='bicubic'),
        T.CenterCrop(384),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    ```

* 模型细节：

    |         Model           |         Model           | Params(M) | FLOPs(G) | Top-1 (%) | Top-5 (%) |
    |:-----------------------:|:-----------------------:|:---------:|:--------:|:---------:|:---------:|
    | DeiT-tiny               |  deit_ti                |  5        |          | 72.2      |  91.1     |
    | DeiT-small              |  deit_s                 | 22        |          | 79.9      |  95.0     |
    | DeiT-base               |  deit_b                 | 86        |          | 81.8      |  95.6     |
    | DeiT-tiny distilled     |  deit_ti_distilled      | 6         |          | 74.5      |  91.9     |
    | DeiT-small distilled    |  deit_s_distilled       | 22        |          | 81.2      |  95.4     |
    | DeiT-base distilled     |  deit_b_distilled       | 87        |          | 83.4      |  96.5     |
    | DeiT-base 384           |  deit_b_384             | 87        |          | 82.9      |  96.2     |
    | DeiT-base distilled 384 |  deit_b_distilled_384   | 88        |          | 85.2      |  97.2     |

## 引用
```
@article{han2020rexnet,
    title = {ReXNet: Diminishing Representational Bottleneck on Convolutional Neural Network},
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
@article{heo2021pit,
    title={Rethinking Spatial Dimensions of Vision Transformers},
    author={Byeongho Heo and Sangdoo Yun and Dongyoon Han and Sanghyuk Chun and Junsuk Choe and Seong Joon Oh},
    journal={arXiv: 2103.16302},
    year={2021},
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