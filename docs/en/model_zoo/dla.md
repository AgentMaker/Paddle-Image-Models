# DLA
* Paper：[Deep Layer Aggregation](https://arxiv.org/abs/1707.06484)
* Origin Repo：[ucbdrive/dla](https://github.com/ucbdrive/dla)
* Code：[dla.py](../../../ppim/models/dla.py)
* Evaluate Transforms：

    ```python
    # backend: pil
    # input_size: 224x224
    transforms = T.Compose([
        T.Resize(256, interpolation='bilinear'),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    ```

* Model Details：

    |         Model         |       Model Name      | Params (M) | FLOPs (G) | Top-1 (%) | Top-5 (%) |   Pretrained Model    |
    |:---------------------:|:---------------------:|:----------:|:---------:|:---------:|:---------:|:---------------------:|
    | DLA-34                | dla_34                | 15.8       | 3.1       | 76.39     |   93.15   | [Download][dla_34]    |
    | DLA-46-c              | dla_46_c              | 1.3        | 0.5       | 64.88     |   86.29   | [Download][dla_46_c]  |
    | DLA-46x-c             | dla_46x_c             | 1.1        | 0.5       | 65.98     |   86.98   | [Download][dla_46x_c] |
    | DLA-60                | dla_60                | 22.0       | 4.2       | 77.02     |   93.31   | [Download][dla_60]    |
    | DLA-60x               | dla_60x               | 17.4       | 3.5       | 78.24     |   94.02   | [Download][dla_60x]   |
    | DLA-60x-c             | dla_60x_c             | 1.3        | 0.6       | 67.91     |   88.43   | [Download][dla_60x_c] |
    | DLA-102               | dla_102               | 33.3       | 7.2       | 79.44     |   94.76   | [Download][dla_102]   |
    | DLA-102x              | dla_102x              | 26.4       | 5.9       | 78.51     |   94.23   | [Download][dla_102x]  |
    | DLA-102x2             | dla_102x2             | 41.4       | 9.3       | 79.45     |   94.64   | [Download][dla_102x2] |
    | DLA-169               | dla_169               | 53.5       | 11.6      | 78.71     |   94.34   | [Download][dla_169]   |


[dla_34]:https://bj.bcebos.com/v1/ai-studio-online/a4e08c790f0247c8ab44cfa9ec6264720a3fab64b51d4ee88d0e7d3511e6348a?responseContentDisposition=attachment%3B%20filename%3Ddla34%2Btricks.pdparams
[dla_46_c]:https://bj.bcebos.com/v1/ai-studio-online/245e16ae6b284b368798a6f8e3cf068e55eea96e22724ec5bff8d146c64da990?responseContentDisposition=attachment%3B%20filename%3Ddla46_c.pdparams
[dla_46x_c]:https://bj.bcebos.com/v1/ai-studio-online/b295201d245247fb8cd601b60919cabf5df51a8997d04380bd07eac71e4152dd?responseContentDisposition=attachment%3B%20filename%3Ddla46x_c.pdparams
[dla_60]:https://bj.bcebos.com/v1/ai-studio-online/e545d431a9f84bb4aecd2c75e34e6169503be2d2e8d246cb9cff393559409f7b?responseContentDisposition=attachment%3B%20filename%3Ddla60.pdparams
[dla_60x]:https://bj.bcebos.com/v1/ai-studio-online/a07ea1cec75a460ebf6dcace4ab0c8c28e923af88dd74573baaaa6db8738168d?responseContentDisposition=attachment%3B%20filename%3Ddla60x.pdparams
[dla_60x_c]:https://bj.bcebos.com/v1/ai-studio-online/0c15f589fa524d1dbe753afe2619f2fe33773c0ca6db4966a3ab8f755fca3c98?responseContentDisposition=attachment%3B%20filename%3Ddla60x_c.pdparams
[dla_102]:https://bj.bcebos.com/v1/ai-studio-online/288ca91946d04df891750eed67b3070ec38a29e9a7b24eff90c0e397d3b82c7f?responseContentDisposition=attachment%3B%20filename%3Ddla102%2Btricks.pdparams
[dla_102x]:https://bj.bcebos.com/v1/ai-studio-online/0653e6aae7594e2a8de94728f6656c375557f7960a8949a1926eb017e978c477?responseContentDisposition=attachment%3B%20filename%3Ddla102x.pdparams
[dla_102x2]:https://bj.bcebos.com/v1/ai-studio-online/80cd37d877974ad18d1ccefdae2a5c2cce1cba2831544deeaea1fa672343cc17?responseContentDisposition=attachment%3B%20filename%3Ddla102x2.pdparams
[dla_169]:https://bj.bcebos.com/v1/ai-studio-online/f299fab9020344d4aee7ccf3a79e98858494e0536bca4703a5f5152747395cca?responseContentDisposition=attachment%3B%20filename%3Ddla169.pdparams
