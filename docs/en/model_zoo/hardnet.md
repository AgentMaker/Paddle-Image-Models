# HarDNet
* Paper：[HarDNet: A Low Memory Traffic Network](https://arxiv.org/abs/1909.00948)
* Origin Repo：[PingoLH/Pytorch-HarDNet](https://github.com/PingoLH/Pytorch-HarDNet)
* Code：[hardnet.py](../../../ppim/models/hardnet.py)
* Evaluate Transforms：

    ```python
    # backend: pil
    # input_size: 224x224
    transforms = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    ```

* Model Details：

    |         Model         |     Model Name        | Params (M) | FLOPs (G) | Top-1 (%) | Top-5 (%) |       Pretrained Model    |
    |:---------------------:|:---------------------:|:----------:|:---------:|:---------:|:---------:|:-------------------------:|
    | HarDNet-68            | hardnet_68            | 17.6       | 4.3       | 76.48     |  93.01    | [Download][hardnet_68]    |
    | HarDNet-85            | hardnet_85            | 36.7       | 9.1       | 78.04     |  93.89    | [Download][hardnet_85]    |
    | HarDNet-39-ds         | hardnet_39_ds         |  3.5       | 0.4       | 72.08     |  90.43    | [Download][hardnet_39_ds] |
    | HarDNet-68-ds         | hardnet_68_ds         |  4.2       | 0.8       | 74.29     |  91.87    | [Download][hardnet_68_ds] |


[hardnet_39_ds]:https://bj.bcebos.com/v1/ai-studio-online/f0b243912f6045bebfe89c65500c4a16534276e45f3544c592713e6e5524ebd2?responseContentDisposition=attachment%3B%20filename%3Dhardnet_39_ds.pdparams
[hardnet_68_ds]:https://bj.bcebos.com/v1/ai-studio-online/a8939896a12243db942263747687cabcad4aae89890345199f1ecfa4fadd6b27?responseContentDisposition=attachment%3B%20filename%3Dhardnet_68_ds.pdparams
[hardnet_68]:https://bj.bcebos.com/v1/ai-studio-online/c82332d24182481db918a848e2ec6d3a6167bd0a96cb4dc1876ce00e224bcb24?responseContentDisposition=attachment%3B%20filename%3Dhardnet_68.pdparams
[hardnet_85]:https://bj.bcebos.com/v1/ai-studio-online/e6f9e798149343968bf80a7ca5e8a7b2e447339202fe451c80878da91895f794?responseContentDisposition=attachment%3B%20filename%3Dhardnet_85.pdparams
