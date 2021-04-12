# CondenseNet V2
* Paper：[CondenseNet V2: Sparse Feature Reactivation for Deep Networks](https://arxiv.org/abs/2104.04382)
* Origin Repo：[jianghaojun/CondenseNetV2](https://github.com/jianghaojun/CondenseNetV2)
* Code：[cdnv2.py](../../../ppim/models/cdnv2.py)
* Evaluate Transforms：

    ```python
    # backend: pil
    # input_size: 224x224
    # models: cdnv2_a and cdnv2_b
    transforms = T.Compose([
        T.Resize(256, interpolation='bilinear'),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # backend: pil
    # input_size: 224x224
    # models: cdnv2_c
    transforms = T.Compose([
        T.Resize(256, interpolation='bicubic'),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    ```

* Model Details：

    |         Model         |       Model Name      | Params (M) | FLOPs (G) | Top-1 (%) | Top-5 (%) |   Pretrained Model    |
    |:---------------------:|:---------------------:|:----------:|:---------:|:---------:|:---------:|:---------------------:|
    |     CondenseNetV2-A   |        cdnv2_a        | 2.0        | 0.05      | 64.37     |   85.15   | [Download][cdnv2_a]   |
    |     CondenseNetV2-B   |        cdnv2_b        | 3.6        | 0.15      | 71.71     |   90.16   | [Download][cdnv2_b]   |
    |     CondenseNetV2-C   |        cdnv2_c        | 6.1        | 0.31      | 75.83     |   92.68   | [Download][cdnv2_c]   |


[cdnv2_a]:https://bj.bcebos.com/v1/ai-studio-online/6ccaae861d004593977e2e3f4d3ad8c9a96e42bbb83347afb58f0d8858abc926?responseContentDisposition=attachment%3B%20filename%3Dcdnv2_a.pdparams
[cdnv2_b]:https://bj.bcebos.com/v1/ai-studio-online/68dbd2a319f34792ae986a0afe6a1db8a1524c0409b4407d8c4c9d699f61d865?responseContentDisposition=attachment%3B%20filename%3Dcdnv2_b.pdparams
[cdnv2_c]:https://bj.bcebos.com/v1/ai-studio-online/d93f60cabe864567b7b8202e614442fbc65b8cbf7ce54b4f98746bc0072832b3?responseContentDisposition=attachment%3B%20filename%3Dcdnv2_c.pdparams


* Citation：

    ```
    @misc{yang2021condensenet,
        title={CondenseNet V2: Sparse Feature Reactivation for Deep Networks}, 
        author={Le Yang and Haojun Jiang and Ruojin Cai and Yulin Wang and Shiji Song and Gao Huang and Qi Tian},
        year={2021},
        eprint={2104.04382},
        archivePrefix={arXiv},
        primaryClass={cs.CV}
    }
    ```
