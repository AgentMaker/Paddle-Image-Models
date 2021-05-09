# Swin Transformer
* Paper：[Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)
* Origin Repo：[microsoft/Swin-Transformer](https://github.com/microsoft/Swin-Transformer)
* Code：[swin.py](../../../ppim/models/swin.py)
* Evaluate Transforms：

    ```python
    # backend: pil
    # input_size: 224x224
    transforms = T.Compose([
        T.Resize(248, interpolation='bicubic'),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # backend: pil
    # input_size: 384x384
    transforms = T.Compose([
        T.Resize((384, 384), interpolation='bicubic'),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    ```

* Model Details：

    |         Model           |       Model Name        | Params (M) | FLOPs (G) | Top-1 (%) | Top-5 (%) |          Pretrained Model        |
    |:-----------------------:|:-----------------------:|:----------:|:---------:|:---------:|:---------:|:--------------------------------:|
    | Swin-tiny               |  swin_ti                | 28         |  4.5      | 81.19     |  95.51    | [Download][swin_ti]              |
    | Swin-small              |  swin_s                 | 50         |  8.7      | 83.18     |  96.24    | [Download][swin_s]               |
    | Swin-base               |  swin_b                 | 88         | 15.4      | 83.42     |  96.45    | [Download][swin_b]               |
    | Swin-base-384           |  swin_b_384             | 88         | 47.1      | 84.47     |  96.95    | [Download][swin_b_384]           |


[swin_ti]:https://bj.bcebos.com/v1/ai-studio-online/19a72dd9eb884f4581492a61fab901e60e858e34569f4805b619eceabd6a4315?responseContentDisposition=attachment%3B%20filename%3Dswin_tiny_patch4_window7_224.pdparams
[swin_s]:https://bj.bcebos.com/v1/ai-studio-online/5a34e4e087824ba48ba9ddd29a22e6fce5d6a2a4a2754e208c61f02090f9d12b?responseContentDisposition=attachment%3B%20filename%3Dswin_small_patch4_window7_224.pdparams
[swin_b]:https://bj.bcebos.com/v1/ai-studio-online/dc2e80e3d4f14880b0700abcb1609c65d541139ab2424b21b6ccdfb64c904a36?responseContentDisposition=attachment%3B%20filename%3Dswin_base_patch4_window7_224.pdparams
[swin_b_384]:https://bj.bcebos.com/v1/ai-studio-online/e013c51b9e134b69933ee7a7c0349be27b8a1a13f823465e8ecbd09fff4aba38?responseContentDisposition=attachment%3B%20filename%3Dswin_base_patch4_window12_384.pdparams


* Citation：

    ```
    @article{liu2021Swin,
    title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
    author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
    journal={arXiv preprint arXiv:2103.14030},
    year={2021}
    }
    ```
