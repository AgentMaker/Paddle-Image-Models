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
from ppim import rexnet_100

# Load the ReXNetV1 x1.0 model
model, val_transforms = rexnet_100(pretrained=True)
```

## Model Zoo
* ReXNet
* RepVGG
* DeiT