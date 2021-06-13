# Paddle-Image-Models
![GitHub forks](https://img.shields.io/github/forks/AgentMaker/Paddle-Image-Models)
![GitHub Repo stars](https://img.shields.io/github/stars/AgentMaker/Paddle-Image-Models)
![Pypi Downloads](https://pepy.tech/badge/ppim)
![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/AgentMaker/Paddle-Image-Models?include_prereleases)
![GitHub](https://img.shields.io/github/license/AgentMaker/Paddle-Image-Models)  

[English](README.md) | 简体中文

一个基于飞桨框架实现的图像预训练模型库。

<table>
  <tbody>
    <tr>
        <td colspan="6" align="center"><b>模型库</b></td>
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
          <li><a href="./docs/cn/model_zoo/dla.md">DLA</a></li>
          <li><a href="./docs/cn/model_zoo/rexnet.md">ReXNet</a></li>
          <li><a href="./docs/cn/model_zoo/rednet.md">RedNet</a></li>
          <li><a href="./docs/cn/model_zoo/repvgg.md">RepVGG</a></li>
          <li><a href="./docs/cn/model_zoo/hardnet.md">HardNet</a></li>
          <li><a href="./docs/cn/model_zoo/cdnv2.md">CondenseNet V2</a></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="./docs/cn/model_zoo/pit.md">PiT</a></li>
          <li><a href="./docs/cn/model_zoo/pvt.md">PvT</a></li>
          <li><a href="./docs/cn/model_zoo/tnt.md">TNT</a></li>
          <li><a href="./docs/cn/model_zoo/deit.md">DeiT</a></li>
          <li><a href="./docs/cn/model_zoo/cait.md">CaiT</a></li>
          <li><a href="./docs/cn/model_zoo/coat.md">CoaT</a></li>
          <li><a href="./docs/cn/model_zoo/levit.md">LeViT</a></li>
          <li><a href="./docs/cn/model_zoo/lvvit.md">LV ViT</a></li>
          <li><a href="./docs/cn/model_zoo/t2t.md">T2T ViT</a></li>
          <li><a href="./docs/cn/model_zoo/swin.md">Swin Transformer</a></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="./docs/cn/model_zoo/mixer.md">MLP-Mixer</a></li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

## 安装
* 通过 pip 进行安装：

    ```shell
    $ pip install ppim
    ```

* 通过 whl 包进行安装：[【Releases Packages】](https://github.com/AgentMaker/Paddle-Image-Models/releases)

## 使用方法
### 快速使用

```python
import paddle
from ppim import rednet_26

# 使用 PPIM whl 包加载模型
model, val_transforms = rednet_26(pretrained=True, return_transforms=True)

# 使用 paddle.hub API 加载模型
# paddlepaddle >= 2.1.0
'''
model, val_transforms = paddle.hub.load(
    'AgentMaker/Paddle-Image-Models:dev', 
    'rednet_26', 
    source='github', 
    force_reload=False, 
    pretrained=True, 
    return_transforms=True
)
'''

# 模型结构总览 
paddle.summary(model, input_size=(1, 3, 224, 224))

# 准备一个随机的输入
x = paddle.randn(shape=(1, 3, 224, 224))

# 模型前向计算
out = model(x)
```

### 分类（PaddleHapi）
    
```python
import paddle
import paddle.nn as nn
import paddle.vision.transforms as T
from paddle.vision import Cifar100

from ppim import rexnet_1_0

# 加载模型
model, val_transforms = rexnet_1_0(pretrained=True, return_transforms=True, class_dim=100)

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