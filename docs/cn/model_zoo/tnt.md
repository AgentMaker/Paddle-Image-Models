# TNT
* 论文：[Transformer in Transformer](https://arxiv.org/abs/2103.00112)
* 官方项目：[huawei-noah/noah-research](https://github.com/huawei-noah/noah-research/tree/master/TNT)
* 模型代码：[tnt.py](../../../ppim/models/tnt.py)
* 验证集数据处理：

    ```python
    # 图像后端：pil
    # 输入图像大小：224x224
    transforms = T.Compose([
        T.Resize(248, interpolation='bicubic'),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    ```

* 模型细节：

    |         Model         |       Model Name      | Params (M) | FLOPs (G) | Top-1 (%) | Top-5 (%) |   Pretrained Model    |
    |:---------------------:|:---------------------:|:----------:|:---------:|:---------:|:---------:|:---------------------:|
    |        TNT-S          |        tnt_s          | 23.8       | 5.2       | 81.53     |   95.74   | [Download][tnt_s]     |


[tnt_s]:https://bj.bcebos.com/v1/ai-studio-online/e8777c29886a47e896f23a26d84917ee51efd05d341a403baed9107160857636?responseContentDisposition=attachment%3B%20filename%3Dtnt_s.pdparams

* 引用：

    ```
    @misc{han2021transformer,
        title={Transformer in Transformer}, 
        author={Kai Han and An Xiao and Enhua Wu and Jianyuan Guo and Chunjing Xu and Yunhe Wang},
        year={2021},
        eprint={2103.00112},
        archivePrefix={arXiv},
        primaryClass={cs.CV}
    }
    ```
