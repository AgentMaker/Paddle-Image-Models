# RedNet
* 论文：[Involution: Inverting the Inherence of Convolution for Visual Recognition](https://arxiv.org/abs/2103.06255)
* 官方项目：[d-li14/involution](https://github.com/d-li14/involution)
* 项目代码：[rednet.py](../../../ppim/models/rednet.py)
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

    |         Model         |     Model Name        | Params (M) | FLOPs (G) | Top-1 (%) | Top-5 (%) |   Pretrained Model     |
    |:---------------------:|:---------------------:|:----------:|:---------:|:---------:|:---------:|:----------------------:|
    | RedNet-26             |     rednet_26         |  9.2       | 1.7       | 75.96     | 93.19     | [Download][rednet_26]  |
    | RedNet-38             |     rednet_38         | 12.4       | 2.2       | 77.48     | 93.57     | [Download][rednet_38]  |
    | RedNet-50             |     rednet_50         | 15.5       | 2.7       | 78.35     | 94.18     | [Download][rednet_50]  |
    | RedNet-101            |     rednet_101        | 25.7       | 4.7       | 78.92     | 94.35     | [Download][rednet_101] |
    | RedNet-152            |     rednet_152        | 34.0       | 6.8       | 79.12     | 94.38     | [Download][rednet_152] |


[rednet_26]:https://bj.bcebos.com/v1/ai-studio-online/14091d6c21774c5fb48d74723db7eaf22e1c5ff621154a588534cb92918c04e2?responseContentDisposition=attachment%3B%20filename%3Drednet26.pdparams
[rednet_38]:https://bj.bcebos.com/v1/ai-studio-online/3c11f732a7804f3d8f6ed2e0cca6da25c2925d841a4d43be8bde60a6d521bf89?responseContentDisposition=attachment%3B%20filename%3Drednet38.pdparams
[rednet_50]:https://bj.bcebos.com/v1/ai-studio-online/084442aeea424f419ce62934bed78af56d0d85d1179146f68dc2ccdf640f8bf3?responseContentDisposition=attachment%3B%20filename%3Drednet50.pdparams
[rednet_101]:https://bj.bcebos.com/v1/ai-studio-online/1527bc759488475981c2daef2f20a13bf181bf55b6b6487691ac0d829873d7df?responseContentDisposition=attachment%3B%20filename%3Drednet101.pdparams
[rednet_152]:https://bj.bcebos.com/v1/ai-studio-online/df78cfc5492541818761fd7f2d8652bffcb6c470c66848949ffd3fc3254ba461?responseContentDisposition=attachment%3B%20filename%3Drednet152.pdparams
