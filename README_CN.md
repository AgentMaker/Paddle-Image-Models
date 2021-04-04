# Paddle-Image-Models
![GitHub forks](https://img.shields.io/github/forks/AgentMaker/Paddle-Image-Models)
![GitHub Repo stars](https://img.shields.io/github/stars/AgentMaker/Paddle-Image-Models)
![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/AgentMaker/Paddle-Image-Models?include_prereleases)
![GitHub](https://img.shields.io/github/license/AgentMaker/Paddle-Image-Models)  

[English](README.md) | 简体中文

一个基于飞桨框架实现的图像预训练模型库。

![](https://ai-studio-static-online.cdn.bcebos.com/34e7bbbc80d24412b3c21efb56778ad43b53f9b1be104e499e0ff8b663a64a53)

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

* [DLA](./docs/cn/model_zoo/dla.md)

* [ReXNet](./docs/cn/model_zoo/rexnet.md)

* [RedNet](./docs/cn/model_zoo/rednet.md)

* [RepVGG](./docs/cn/model_zoo/repvgg.md)

* [HarDNet](./docs/cn/model_zoo/hardnet.md)

* [PiT](./docs/cn/model_zoo/pit.md)

* [TNT](./docs/cn/model_zoo/tnt.md)

* [DeiT](./docs/cn/model_zoo/deit.md)

## 引用
```
@article{han2020rexnet,
    title = {ReXNet: Diminishing Representational Bottleneck on Convolutional Neural Network},
    author = {Han, Dongyoon and Yun, Sangdoo and Heo, Byeongho and Yoo, YoungJoon},
    journal = {arXiv preprint arXiv:2007.00992},
    year = {2020},
}

@InProceedings{Li_2021_CVPR,
    title = {Involution: Inverting the Inherence of Convolution for Visual Recognition},
    author = {Li, Duo and Hu, Jie and Wang, Changhu and Li, Xiangtai and She, Qi and Zhu, Lei and Zhang, Tong and Chen, Qifeng},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2021}
}

@article{ding2021repvgg,
    title={RepVGG: Making VGG-style ConvNets Great Again},
    author={Ding, Xiaohan and Zhang, Xiangyu and Ma, Ningning and Han, Jungong and Ding, Guiguang and Sun, Jian},
    journal={arXiv preprint arXiv:2101.03697},
    year={2021}
}

@article{heo2021pit,
    title={Rethinking Spatial Dimensions of Vision Transformers},
    author={Byeongho Heo and Sangdoo Yun and Dongyoon Han and Sanghyuk Chun and Junsuk Choe and Seong Joon Oh},
    journal={arXiv: 2103.16302},
    year={2021},
}

@article{touvron2020deit,
    title = {Training data-efficient image transformers & distillation through attention},
    author = {Hugo Touvron and Matthieu Cord and Matthijs Douze and Francisco Massa and Alexandre Sablayrolles and Herv'e J'egou},
    journal = {arXiv preprint arXiv:2012.12877},
    year = {2020}
}

@misc{han2021transformer,
    title={Transformer in Transformer}, 
    author={Kai Han and An Xiao and Enhua Wu and Jianyuan Guo and Chunjing Xu and Yunhe Wang},
    year={2021},
    eprint={2103.00112},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}

@misc{chao2019hardnet,
    title={HarDNet: A Low Memory Traffic Network}, 
    author={Ping Chao and Chao-Yang Kao and Yu-Shan Ruan and Chien-Hsiang Huang and Youn-Long Lin},
    year={2019},
    eprint={1909.00948},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}

@misc{yu2019deep,
    title={Deep Layer Aggregation}, 
    author={Fisher Yu and Dequan Wang and Evan Shelhamer and Trevor Darrell},
    year={2019},
    eprint={1707.06484},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}

@misc{dosovitskiy2020image,
    title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale}, 
    author={Alexey Dosovitskiy and Lucas Beyer and Alexander Kolesnikov and Dirk Weissenborn and Xiaohua Zhai and Thomas Unterthiner and Mostafa Dehghani and Matthias Minderer and Georg Heigold and Sylvain Gelly and Jakob Uszkoreit and Neil Houlsby},
    year={2020},
    eprint={2010.11929},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```