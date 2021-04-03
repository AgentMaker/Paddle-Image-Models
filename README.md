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
    $ pip install ppim==1.0.5 -i https://pypi.python.org/pypi 
    ```

* Install by wheel package：[【Releases Packages】](https://github.com/AgentMaker/Paddle-Image-Models/releases)

## Usage
* Quick Start

    ```python
    import paddle
    from ppim import rednet_26

    # Load the model
    model, val_transforms = rednet_26(pretrained=True)

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

    from ppim import rexnet_1_0

    # Load the model
    model, val_transforms = rexnet_1_0(pretrained=True)

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
* [DLA](./docs/en/model_zoo/dla.md)

* [ReXNet](./docs/en/model_zoo/rexnet.md)

* [RedNet](./docs/en/model_zoo/rednet.md)

* [RepVGG](./docs/en/model_zoo/repvgg.md)

* [HarDNet](./docs/en/model_zoo/hardnet.md)

* [PiT](./docs/en/model_zoo/pit.md)

* [TNT](./docs/en/model_zoo/tnt.md)

* [DeiT](./docs/en/model_zoo/deit.md)

## Citation
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
```
@misc{han2021transformer,
      title={Transformer in Transformer}, 
      author={Kai Han and An Xiao and Enhua Wu and Jianyuan Guo and Chunjing Xu and Yunhe Wang},
      year={2021},
      eprint={2103.00112},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```