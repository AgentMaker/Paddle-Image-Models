# TNT
* Paper：[Transformer in Transformer](https://arxiv.org/abs/2103.00112)
* Origin Repo：[huawei-noah/noah-research](https://github.com/huawei-noah/noah-research/tree/master/TNT)
* Code：[tnt.py](../../../ppim/models/tnt.py)
* Evaluate Transforms：

    ```python
    # backend: pil
    # input_size: 224x224
    transforms = T.Compose([
        T.Resize(248, interpolation='bicubic'),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    ```

* Model Details：

    |         Model         |       Model Name      | Params (M) | FLOPs (G) | Top-1 (%) | Top-5 (%) |   Pretrained Model    |
    |:---------------------:|:---------------------:|:----------:|:---------:|:---------:|:---------:|:---------------------:|
    |        TNT-S          |        tnt_s          | 23.8       | 5.2       | 81.53     |   95.74   | [Download][tnt_s]     |


[tnt_s]:https://bj.bcebos.com/v1/ai-studio-online/e8777c29886a47e896f23a26d84917ee51efd05d341a403baed9107160857636?responseContentDisposition=attachment%3B%20filename%3Dtnt_s.pdparams

* Citation：

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
