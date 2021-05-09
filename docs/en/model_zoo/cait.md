# CaiT
* Paper：[Going deeper with Image Transformers](https://arxiv.org/abs/2103.17239)
* Origin Repo：[facebookresearch/deit](https://github.com/facebookresearch/deit)
* Code：[cait.py](../../../ppim/models/cait.py)
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
        T.Resize(384, interpolation='bicubic'),
        T.CenterCrop(384),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # backend: pil
    # input_size: 448x448
    transforms = T.Compose([
        T.Resize(448, interpolation='bicubic'),
        T.CenterCrop(448),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    ```

* Model Details：

    |         Model           |       Model Name        | Params (M) | FLOPs (G) | Top-1 (%) | Top-5 (%) |          Pretrained Model        |
    |:-----------------------:|:-----------------------:|:----------:|:---------:|:---------:|:---------:|:--------------------------------:|
    | CaiT-xxs-24             |  cait_xxs_24            | 12.0       |  2.5      | 78.50     |  94.30    | [Download][cait_xxs_24]          |
    | CaiT-xxs-36             |  cait_xxs_36            | 17.3       |  3.8      | 79.87     |  94.90    | [Download][cait_xxs_36]          |
    | CaiT-s-24               |  cait_s_24              | 49.6       |  9.4      | 83.40     |  96.62    | [Download][cait_s_24]            |
    | CaiT-xxs-24-384         |  cait_xxs_24_384        | 12.0       |  9.5      | 80.97     |  95.64    | [Download][cait_xxs_24_384]      |
    | CaiT-xxs-36-384         |  cait_xxs_36_384        | 17.3       | 14.2      | 82.20     |  96.15    | [Download][cait_xxs_36_384]      |
    | CaiT-xs-24-384          |  cait_xs_24_384         | 26.6       | 19.3      | 84.06     |  96.89    | [Download][cait_xs_24_384]       |
    | CaiT-s-24-384           |  cait_s_24_384          | 49.6       | 32.2      | 85.05     |  97.34    | [Download][cait_s_24_384]        |
    | CaiT-s-36-384           |  cait_s_36_384          | 68.2       | 48.0      | 85.45     |  97.48    | [Download][cait_s_36_384]        |
    | CaiT-m-36-384           |  cait_m_36_384          | 270.9      | 173.3     | 86.06     |  97.73    | [Download][cait_m_36_384]        |
    | CaiT-m-48-448           |  cait_m_48_448          | 356.0      | 329.6     | 86.49     |  97.75    | [Download][cait_m_48_448]        |


[cait_xxs_24]:https://bj.bcebos.com/v1/ai-studio-online/f104732e10e64c48b2848a78b7fa5db45d27a8eed0754c04b367d0708e7242ea?responseContentDisposition=attachment%3B%20filename%3DCaiT_XXS24_224.pdparams
[cait_xxs_36]:https://bj.bcebos.com/v1/ai-studio-online/af39ff4c2d6a48faa6dfb901b4fc1de4ae082d767bdc4deb824ae3b600823f1e?responseContentDisposition=attachment%3B%20filename%3DCaiT_XXS36_224.pdparams
[cait_s_24]:https://bj.bcebos.com/v1/ai-studio-online/4ecc9cecc89d43cbacf68a0ba14d58a1c9311cc86da3426ab5674fd79827a89a?responseContentDisposition=attachment%3B%20filename%3DCaiT_S24_224.pdparams
[cait_xxs_24_384]:https://bj.bcebos.com/v1/ai-studio-online/0e3615fb421a4301b08fcd675e063a101f4962bad59649f498912123aa0454a4?responseContentDisposition=attachment%3B%20filename%3DCaiT_XXS24_384.pdparams
[cait_xxs_36_384]:https://bj.bcebos.com/v1/ai-studio-online/b9f2db8a9c1c43ed971ea4779361c213512ef4c25b664216ab151b6ea60260a7?responseContentDisposition=attachment%3B%20filename%3DCaiT_XXS36_384.pdparams
[cait_xs_24_384]:https://bj.bcebos.com/v1/ai-studio-online/b36139e3caa4427eaaf51aa6de33c8b21f209eef97a44aacb4ec4fe136f93d85?responseContentDisposition=attachment%3B%20filename%3DCaiT_XS24_384.pdparams
[cait_s_24_384]:https://bj.bcebos.com/v1/ai-studio-online/4f57d1db346e435ebb81567399668d6181f054353f6c47e89e9f109b33d724c1?responseContentDisposition=attachment%3B%20filename%3DCaiT_S24_384.pdparams
[cait_s_36_384]:https://bj.bcebos.com/v1/ai-studio-online/445e36df9ec54b23a348bf977b81d92c6f54b31fb28b454d8742e056f99e6417?responseContentDisposition=attachment%3B%20filename%3DCaiT_S36_384.pdparams
[cait_m_36_384]:https://bj.bcebos.com/v1/ai-studio-online/4c73e395068747b9b5c8cdafc3d1b6122a7ed94e6e74481e836eb38c8c46a6eb?responseContentDisposition=attachment%3B%20filename%3DCaiT_M36_384.pdparams
[cait_m_48_448]:https://bj.bcebos.com/v1/ai-studio-online/70515fadc26f48d4b98b33304d8de7c7b955086688324aec8100e5df8a66b15d?responseContentDisposition=attachment%3B%20filename%3DCaiT_M48_448.pdparams


* Citation：

    ```
    @article{touvron2021cait,
    title={Going deeper with Image Transformers},
    author={Hugo Touvron and Matthieu Cord and Alexandre Sablayrolles and Gabriel Synnaeve and Herv'e J'egou},
    journal={arXiv preprint arXiv:2103.17239},
    year={2021}
    }
    ```
