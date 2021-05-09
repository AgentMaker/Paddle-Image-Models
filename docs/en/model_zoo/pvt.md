# PVT
* Paper：[Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions](https://arxiv.org/abs/2102.12122)
* Origin Repo：[whai362/PVT](https://github.com/whai362/PVT)
* Code：[pvt.py](../../../ppim/models/pvt.py)
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
    ```

* Model Details：

    |         Model         |     Model Name        | Params (M) | FLOPs (G) | Top-1 (%) | Top-5 (%) |      Pretrained Model        |
    |:---------------------:|:---------------------:|:----------:|:---------:|:---------:|:---------:|:----------------------------:|
    | PVT-Tiny              | pvt_ti                | 13.2       | 1.9       | 74.96     |  92.47    | [Download][pvt_ti]           |
    | PVT-Small             | pvt_s                 | 24.5       | 3.8       | 79.87     |  95.05    | [Download][pvt_s]            |
    | PVT-Medium            | pvt_m                 | 44.2       | 6.7       | 81.48     |  95.75    | [Download][pvt_m]            |
    | PVT-Large             | pvt_l                 | 61.4       | 9.8       | 81.74     |  95.87    | [Download][pvt_l]            |


[pvt_ti]:https://bj.bcebos.com/v1/ai-studio-online/f833d36454ae4c11be0f5d2eb3041a7e9c2df10b8518434193c0b7c8853dfddf?responseContentDisposition=attachment%3B%20filename%3Dpvt_tiny.pdparams
[pvt_s]:https://bj.bcebos.com/v1/ai-studio-online/608703b1387b44a78d01f09f0c572bd163edecf2354243dda1afeab2b58abb06?responseContentDisposition=attachment%3B%20filename%3Dpvt_small.pdparams
[pvt_m]:https://bj.bcebos.com/v1/ai-studio-online/232d73f40a3b45bb96786a8ae6a58f93967ada580a354266910bb63caa96201b?responseContentDisposition=attachment%3B%20filename%3Dpvt_medium.pdparams
[pvt_l]:https://bj.bcebos.com/v1/ai-studio-online/08b2064702304e13893337d1b1017941ced31fc4f7c644acb4a44a1a81c66e55?responseContentDisposition=attachment%3B%20filename%3Dpvt_large.pdparams


* Citation：

    ```
    @misc{wang2021pyramid,
        title={Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions}, 
        author={Wenhai Wang and Enze Xie and Xiang Li and Deng-Ping Fan and Kaitao Song and Ding Liang and Tong Lu and Ping Luo and Ling Shao},
        year={2021},
        eprint={2102.12122},
        archivePrefix={arXiv},
        primaryClass={cs.CV}
    }
    ```
