# Paddle-Image-Models
![GitHub forks](https://img.shields.io/github/forks/AgentMaker/Paddle-Image-Models)
![GitHub Repo stars](https://img.shields.io/github/stars/AgentMaker/Paddle-Image-Models)
![Pypi Downloads](https://pepy.tech/badge/ppim)
![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/AgentMaker/Paddle-Image-Models?include_prereleases)
![GitHub](https://img.shields.io/github/license/AgentMaker/Paddle-Image-Models)  

English | [简体中文](README_CN.md)

A PaddlePaddle version image model zoo.

![](https://ai-studio-static-online.cdn.bcebos.com/34e7bbbc80d24412b3c21efb56778ad43b53f9b1be104e499e0ff8b663a64a53)

## Install Package
* Install by pip：

    ```shell
    $ pip install ppim
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
    model, val_transforms = rexnet_1_0(pretrained=True, class_dim=100)

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

* Segmentation

    * PaddleSeg x PPIM

        ```yaml
        # config
        ...

        model:
        backbone:
            type: rexnet_1_0 # PPIM model name
            pretrained: True # If load the pretrained model
            get_features: True # Get image features for segmentation
        
        ...
        ```

    * Train script：[train.py](./tools/seg/train.py)
    
        ```python
        # train.py
        ...
        
        '''
            Add the ppim models.
        '''
        import ppim.models as models
        from inspect import isfunction

        for model in models.__dict__.values():
            if isfunction(model):
                manager.BACKBONES.add_component(model)

        ...
        ```


## Model Zoo

* [DLA](./docs/en/model_zoo/dla.md)

* [ReXNet](./docs/en/model_zoo/rexnet.md)

* [RedNet](./docs/en/model_zoo/rednet.md)

* [RepVGG](./docs/en/model_zoo/repvgg.md)

* [HarDNet](./docs/en/model_zoo/hardnet.md)

* [CondenseNet V2](./docs/en/model_zoo/cdnv2.md)

* [PiT](./docs/en/model_zoo/pit.md)

* [PVT](./docs/en/model_zoo/pvt.md)

* [TNT](./docs/en/model_zoo/tnt.md)

* [DeiT](./docs/en/model_zoo/deit.md)

* [CaiT](./docs/en/model_zoo/cait.md)

* [Swin Transformer](./docs/en/model_zoo/swin.md)
