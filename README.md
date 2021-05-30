# Paddle-Image-Models
![GitHub forks](https://img.shields.io/github/forks/AgentMaker/Paddle-Image-Models)
![GitHub Repo stars](https://img.shields.io/github/stars/AgentMaker/Paddle-Image-Models)
![Pypi Downloads](https://pepy.tech/badge/ppim)
![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/AgentMaker/Paddle-Image-Models?include_prereleases)
![GitHub](https://img.shields.io/github/license/AgentMaker/Paddle-Image-Models)  

English | [简体中文](README_CN.md)

A PaddlePaddle version image model zoo.

<table>
  <tbody>
    <tr>
        <td colspan="6" align="center"><b>Model Zoo</b></td>
    </tr>
    <tr align="center" valign="bottom">
      <td>
        <b>CNN</b>
      </td>
      <td>
        <b>Transformer</b>
      </td>
      <td>
        <b>MLP</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
          <li><a href="./docs/en/model_zoo/dla.md">DLA</a></li>
          <li><a href="./docs/en/model_zoo/rexnet.md">ReXNet</a></li>
          <li><a href="./docs/en/model_zoo/rednet.md">RedNet</a></li>
          <li><a href="./docs/en/model_zoo/repvgg.md">RepVGG</a></li>
          <li><a href="./docs/en/model_zoo/hardnet.md">HardNet</a></li>
          <li><a href="./docs/en/model_zoo/cdnv2.md">CondenseNet V2</a></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="./docs/en/model_zoo/pit.md">PiT</a></li>
          <li><a href="./docs/en/model_zoo/pvt.md">PvT</a></li>
          <li><a href="./docs/en/model_zoo/tnt.md">TNT</a></li>
          <li><a href="./docs/en/model_zoo/deit.md">DeiT</a></li>
          <li><a href="./docs/en/model_zoo/cait.md">CaiT</a></li>
          <li><a href="./docs/en/model_zoo/coat.md">CoaT</a></li>
          <li><a href="./docs/en/model_zoo/levit.md">LeViT</a></li>
          <li><a href="./docs/en/model_zoo/lvvit.md">LV ViT</a></li>
          <li><a href="./docs/en/model_zoo/swin.md">Swin Transformer</a></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="./docs/en/model_zoo/mixer.md">MLP-Mixer</a></li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

## Install Package
* Install by pip：

    ```shell
    $ pip install ppim
    ```

* Install by wheel package：[【Releases Packages】](https://github.com/AgentMaker/Paddle-Image-Models/releases)

## Usage
### Quick Start

```python
import paddle
from ppim import rednet_26

# Load the model with PPIM wheel package
model, val_transforms = rednet_26(pretrained=True, return_transforms=True)

# Load the model with paddle.hub API
# paddlepaddle >= 2.1.0
'''
model, val_transforms = paddle.hub.load(
    'AgentMaker/Paddle-Image-Models:release', 
    'rednet_26', 
    source='github', 
    force_reload=False, 
    pretrained=True, 
    return_transforms=True
)
'''

# Model summary 
paddle.summary(model, input_size=(1, 3, 224, 224))

# Random a input
x = paddle.randn(shape=(1, 3, 224, 224))

# Model forword
out = model(x)
```

### Classification（PaddleHapi）
    
```python
import paddle
import paddle.nn as nn
import paddle.vision.transforms as T
from paddle.vision import Cifar100

from ppim import rexnet_1_0

# Load the model
model, val_transforms = rexnet_1_0(pretrained=True, return_transforms=True, class_dim=100)

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

### Segmentation（PaddleSeg）

|   Supported Models      |
|:-----------------------:|
| ReXNet                  |
| RedNet                  |
| RepVGG                  |
| CondenseNet V2          |
| Coming soon...          |

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
