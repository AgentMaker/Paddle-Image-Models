# LV ViT
* 论文：[Token Labeling: Training a 85.4% Top-1 Accuracy Vision Transformer with 56M Parameters on ImageNet](https://arxiv.org/abs/2104.10858)
* 官方项目：[zihangJiang/TokenLabeling](https://github.com/zihangJiang/TokenLabeling)
* 模型代码：[lvvit.py](../../../ppim/models/lvvit.py)
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

    # 图像后端：pil
    # 输入图像大小：448x448
    transforms = T.Compose([
        T.Resize(448, interpolation='bicubic'),
        T.CenterCrop(448),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    ```

* 模型细节：

    |         Model           |       Model Name        | Params (M) | FLOPs (G) | Top-1 (%) | Top-5 (%) |          Pretrained Model        |
    |:-----------------------:|:-----------------------:|:----------:|:---------:|:---------:|:---------:|:--------------------------------:|
    | LV-ViT-S                |  lvvit_s                | 26.2       |  6.6      | 83.17     |  95.87    | [Download][lvvit_s]              |
    | LV-ViT-M                |  lvvit_m                | 55.8       | 16.0      | 83.88     |  96.05    | [Download][lvvit_m]              |
    | LV-ViT-S-384            |  lvvit_s_384            | 26.3       | 22.2      | 84.56     |  96.39    | [Download][lvvit_s_384]          |
    | LV-ViT-M-384            |  lvvit_m_384            | 56.0       | 42.2      | 85.34     |  96.72    | [Download][lvvit_m_384]          |
    | LV-ViT-M-448            |  lvvit_m_448            | 56.1       | 61.0      | 85.47     |  96.82    | [Download][lvvit_m_448]          |
    | LV-ViT-L-448            |  lvvit_l_448            | 150.5      | 157.2     | 86.09     |  96.85    | [Download][lvvit_l_448]          |


[lvvit_s]:https://bj.bcebos.com/v1/ai-studio-online/bf798145d3094d4ab89f99d87a3f99ad576361f3e05e46f4a622de90ef565e9b?responseContentDisposition=attachment%3B%20filename%3Dlvvit_s_224.pdparams
[lvvit_m]:https://bj.bcebos.com/v1/ai-studio-online/c34bcd65d1c94089ab269ffb8927133a7fab39c6a0c44dca8e1c995155cabcd0?responseContentDisposition=attachment%3B%20filename%3Dlvvit_m_224.pdparams
[lvvit_s_384]:https://bj.bcebos.com/v1/ai-studio-online/aa4fa51138ea41cb9b413db1308ccc01319f896413764a2d9a3b6e6a23da1ade?responseContentDisposition=attachment%3B%20filename%3Dlvvit_s_384.pdparams
[lvvit_m_384]:https://bj.bcebos.com/v1/ai-studio-online/97d6a53daf55477bbf6e386e00d4763157bcbcea295b402ebb3a26725eaeb772?responseContentDisposition=attachment%3B%20filename%3Dlvvit_m_384.pdparams
[lvvit_m_448]:https://bj.bcebos.com/v1/ai-studio-online/b83be46049ac44cfb0821f429e54621020e815f8019944dca81e73a6736b0fdf?responseContentDisposition=attachment%3B%20filename%3Dlvvit_m_448.pdparams
[lvvit_l_448]:https://bj.bcebos.com/v1/ai-studio-online/abd5019da732445eae48ed4eaeff874fc2c00d8d43934ff783d77720b09faef8?responseContentDisposition=attachment%3B%20filename%3Dlvvit_l_448.pdparams


* 引用：

    ```
    @article{jiang2021token,
    title={Token Labeling: Training a 85.5% Top-1 Accuracy Vision Transformer with 56M Parameters on ImageNet},
    author={Jiang, Zihang and Hou, Qibin and Yuan, Li and Zhou, Daquan and Jin, Xiaojie and Wang, Anran and Feng, Jiashi},
    journal={arXiv preprint arXiv:2104.10858},
    year={2021}
    }
    ```
