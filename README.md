# Paddle-Image-Models
![GitHub forks](https://img.shields.io/github/forks/AgentMaker/Paddle-Image-Models)
![GitHub Repo stars](https://img.shields.io/github/stars/AgentMaker/Paddle-Image-Models)
![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/AgentMaker/Paddle-Image-Models?include_prereleases)
![GitHub](https://img.shields.io/github/license/AgentMaker/Paddle-Image-Models)  

A PaddlePaddle version image model zoo.

## Install Package
* Install by pip：
```shell
$ pip install ppim==1.0.0 -i https://pypi.python.org/pypi 
```
* Install by wheel package：[【Releases Packages】](https://github.com/AgentMaker/Paddle-Image-Models/releases)

## Quick Start
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
* RedNet
* RepVGG
* DeiT
