# DeiT
* 论文：[Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877)
* 官方项目：[facebookresearch/deit](https://github.com/facebookresearch/deit)
* 模型代码：[deit.py](../../../ppim/models/deit.py)
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
    # 输入图像大小：384X384
    transforms = T.Compose([
        T.Resize(384, interpolation='bicubic'),
        T.CenterCrop(384),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    ```

* 模型细节：

    |         Model           |       Model Name        | Params (M) | FLOPs (G) | Top-1 (%) | Top-5 (%) |          Pretrained Model        |
    |:-----------------------:|:-----------------------:|:----------:|:---------:|:---------:|:---------:|:--------------------------------:|
    | DeiT-tiny               |  deit_ti                |  5.7       |  1.1      | 72.18     |  91.11    | [Download][deit_ti]              |
    | DeiT-small              |  deit_s                 | 22.0       |  4.2      | 79.85     |  95.04    | [Download][deit_s]               |
    | DeiT-base               |  deit_b                 | 86.4       | 16.8      | 81.99     |  95.74    | [Download][deit_b]               |
    | DeiT-tiny distilled     |  deit_ti_distilled      |  5.9       |  1.1      | 74.50     |  91.89    | [Download][deit_ti_distilled]    |
    | DeiT-small distilled    |  deit_s_distilled       | 22.4       |  4.3      | 81.22     |  95.39    | [Download][deit_s_distilled]     |
    | DeiT-base distilled     |  deit_b_distilled       | 87.2       | 16.9      | 83.39     |  96.49    | [Download][deit_b_distilled]     |
    | DeiT-base 384           |  deit_b_384             | 86.4       | 49.3      | 83.10     |  96.37    | [Download][deit_b_384]           |
    | DeiT-base distilled 384 |  deit_b_distilled_384   | 87.2       | 49.4      | 85.43     |  97.33    | [Download][deit_b_distilled_384] |


[deit_ti]:https://bj.bcebos.com/v1/ai-studio-online/1e91e6ab967b4b0f9940891c6f77f98ca612d5a767b8482498c364c11d65b44b?responseContentDisposition=attachment%3B%20filename%3DDeiT_tiny_patch16_224.pdparams
[deit_s]:https://bj.bcebos.com/v1/ai-studio-online/56fb3b56543d495aa36cc244e8f25e3e321747cfcedd48c28830ea3a22f4a82a?responseContentDisposition=attachment%3B%20filename%3DDeiT_small_patch16_224.pdparams
[deit_b]:https://bj.bcebos.com/v1/ai-studio-online/38be4cdffc0240c18e9e4905641e9e8171277f42646947e5b3dbcd68c59a6d81?responseContentDisposition=attachment%3B%20filename%3DDeiT_base_patch16_224.pdparams
[deit_ti_distilled]:https://bj.bcebos.com/v1/ai-studio-online/dd0ff3e26c1e4fd4b56698a43a62febd35bdc8153563435b898cdd9480cd8720?responseContentDisposition=attachment%3B%20filename%3DDeiT_tiny_distilled_patch16_224.pdparams
[deit_s_distilled]:https://bj.bcebos.com/v1/ai-studio-online/5ab1d5f92e1f44d39db09ab2233143f8fd27788c9b4f46bd9f1d5f2cb760933e?responseContentDisposition=attachment%3B%20filename%3DDeiT_small_distilled_patch16_224.pdparams
[deit_b_distilled]:https://bj.bcebos.com/v1/ai-studio-online/24692c628ab64bfc9bb72fc8a5b3d209080b5ad94227472f98d3bb7cb6732e67?responseContentDisposition=attachment%3B%20filename%3DDeiT_base_distilled_patch16_224.pdparams
[deit_b_384]:https://bj.bcebos.com/v1/ai-studio-online/de491e7155e94ac2b13b2a97e432155ed6d502e8a0114e4e90ffd6ce9dce63cc?responseContentDisposition=attachment%3B%20filename%3DDeiT_base_patch16_384.pdparams
[deit_b_distilled_384]:https://bj.bcebos.com/v1/ai-studio-online/0a84b9ea45d0412d9bebae9ea3404e679221c3d0c8e542bf9d6a64f810983b25?responseContentDisposition=attachment%3B%20filename%3DDeiT_base_distilled_patch16_384.pdparams

* 引用：

    ```
    @article{touvron2020deit,
        title = {Training data-efficient image transformers & distillation through attention},
        author = {Hugo Touvron and Matthieu Cord and Matthijs Douze and Francisco Massa and Alexandre Sablayrolles and Herv'e J'egou},
        journal = {arXiv preprint arXiv:2012.12877},
        year = {2020}
    }
    ```
