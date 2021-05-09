# ReXNet
* 论文：[ReXNet: Diminishing Representational Bottleneck on Convolutional Neural Network](https://arxiv.org/abs/2007.00992)
* 官方项目：[clovaai/rexnet](https://github.com/clovaai/rexnet)
* 模型代码：[rexnet.py](../../../ppim/models/rexnet.py)
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

    |         Model         |     Model Name        | Params (M) | FLOPs (G) | Top-1 (%) | Top-5 (%) |    Pretrained Model    |
    |:---------------------:|:---------------------:|:----------:|:---------:|:---------:|:---------:|:----------------------:|
    | ReXNet-1.0            | rexnet_1_0            |  4.8       | 0.4       | 77.86     | 93.87     | [Download][rexnet_1_0] |
    | ReXNet-1.3            | rexnet_1_3            |  7.6       | 0.7       | 79.50     | 94.68     | [Download][rexnet_1_3] |
    | ReXNet-1.5            | rexnet_1_5            |  7.6       | 0.7       | 80.32     | 95.17     | [Download][rexnet_1_5] |
    | ReXNet-2.0            | rexnet_2_0            |  16.0      | 1.5       | 81.64     | 95.66     | [Download][rexnet_2_0] |
    | ReXNet-3.0            | rexnet_3_0            |  34.0      | 3.4       | 82.45     | 96.26     | [Download][rexnet_3_0] |


[rexnet_1_0]:https://bj.bcebos.com/v1/ai-studio-online/6c890dd95dfc4e388335adfa298163d3ab413cca558e4abe966d52cb5c3aee31?responseContentDisposition=attachment%3B%20filename%3Drexnetv1_1.0x.pdparams
[rexnet_1_3]:https://bj.bcebos.com/v1/ai-studio-online/41a4cc3e6d9545b9b69b4782cafa01147eb7661ec6af4f43841adc734149b3a7?responseContentDisposition=attachment%3B%20filename%3Drexnetv1_1.3x.pdparams
[rexnet_1_5]:https://bj.bcebos.com/v1/ai-studio-online/20b131a7cb1840b5aed37c512b2665fb20c72eebe4344da5a3c6f0ab0592a323?responseContentDisposition=attachment%3B%20filename%3Drexnetv1_1.5x.pdparams
[rexnet_2_0]:https://bj.bcebos.com/v1/ai-studio-online/b4df9f7be43446b0952a25ee6e83f2e443e3b879a00046f6bb33278319cb5fd0?responseContentDisposition=attachment%3B%20filename%3Drexnetv1_2.0x.pdparams
[rexnet_3_0]:https://bj.bcebos.com/v1/ai-studio-online/9663f0570f0a4e4a8dde0b9799c539f5e22f46917d3d4e5a9d566cd213032d25?responseContentDisposition=attachment%3B%20filename%3Drexnetv1_3.0x.pdparams


* 引用：

    ```
    @article{han2020rexnet,
        title = {ReXNet: Diminishing Representational Bottleneck on Convolutional Neural Network},
        author = {Han, Dongyoon and Yun, Sangdoo and Heo, Byeongho and Yoo, YoungJoon},
        journal = {arXiv preprint arXiv:2007.00992},
        year = {2020},
    }
    ```
