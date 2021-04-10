# RepVGG
* Paper：[RepVGG: Making VGG-style ConvNets Great Again](https://arxiv.org/abs/2101.03697)
* Origin Repo：[DingXiaoH/RepVGG](https://github.com/DingXiaoH/RepVGG)
* Code：[repvgg.py](../../../ppim/models/repvgg.py)
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

    |         Model         |     Model Name        | Params (M) | FLOPs (G) | Top-1 (%) | Top-5 (%) |     Pretrained Model    |
    |:---------------------:|:---------------------:|:----------:|:---------:|:---------:|:---------:|:-----------------------:|
    | RepVGG-A0             | repvgg_a0             |  8.3       | 1.4       | 72.42     |  90.49    | [Download][repvgg_a0]   |
    | RepVGG-A1             | repvgg_a1             | 12.8       | 2.4       | 74.46     |  91.85    | [Download][repvgg_a1]   |
    | RepVGG-A2             | repvgg_a2             | 25.5       | 5.1       | 76.46     |  93.00    | [Download][repvgg_a2]   |
    | RepVGG-B0             | repvgg_b0             | 14.3       | 3.1       | 75.15     |  92.42    | [Download][repvgg_b0]   |
    | RepVGG-B1             | repvgg_b1             | 51.8       | 11.8      | 78.37     |  94.10    | [Download][repvgg_b1]   |
    | RepVGG-B2             | repvgg_b2             | 80.3       | 18.4      | 78.79     |  94.42    | [Download][repvgg_b2]   |
    | RepVGG-B3             | repvgg_b3             | 111.0      | 26.2      | 80.50     |  95.26    | [Download][repvgg_b3]   |
    | RepVGG-B1g2           | repvgg_b1g2           | 41.4       | 8.8       | 77.80     |  93.88    | [Download][repvgg_b1g2] |
    | RepVGG-B1g4           | repvgg_b1g4           | 36.1       | 7.3       | 77.58     |  93.84    | [Download][repvgg_b1g4] |
    | RepVGG-B2g4           | repvgg_b2g4           | 55.8       | 11.3      | 79.38     |  94.68    | [Download][repvgg_b2g4] |
    | RepVGG-B3g4           | repvgg_b3g4           | 75.6       | 16.1      | 80.21     |  95.09    | [Download][repvgg_b3g4] |


[repvgg_a0]:https://bj.bcebos.com/v1/ai-studio-online/26d1d26e0d0141deafeb7e9980ec8b5a555232b938e44fefa93da930422af42b?responseContentDisposition=attachment%3B%20filename%3DRepVGG_A0.pdparams
[repvgg_a1]:https://bj.bcebos.com/v1/ai-studio-online/afa4629fb917427a829bb278250b84b0380d580b40fc4e478eb5fdb75fe22096?responseContentDisposition=attachment%3B%20filename%3DRepVGG_A1.pdparams
[repvgg_a2]:https://bj.bcebos.com/v1/ai-studio-online/200f4d6038834fd49796941f5acf65308e6e096d2b8c496abb9d1c0204f44cb1?responseContentDisposition=attachment%3B%20filename%3DRepVGG_A2.pdparams
[repvgg_b0]:https://bj.bcebos.com/v1/ai-studio-online/93c345b4a76b4f88b3590fa703a270b009cc9c05481640a49e8654222459e79f?responseContentDisposition=attachment%3B%20filename%3DRepVGG_B0.pdparams
[repvgg_b1]:https://bj.bcebos.com/v1/ai-studio-online/b2f8171754bd4d3cb44739b675dc1f0b8cb77ebefdad47ec82ce98292726bf2c?responseContentDisposition=attachment%3B%20filename%3DRepVGG_B1.pdparams
[repvgg_b2]:https://bj.bcebos.com/v1/ai-studio-online/9fc65aab46b441dca194f974bdf420710b2144e941704330869d62a2ab9cb0b6?responseContentDisposition=attachment%3B%20filename%3DRepVGG_B2.pdparams
[repvgg_b3]:https://bj.bcebos.com/v1/ai-studio-online/8d902ba9ebf3441e896e8d7078544005a0715ca6867f4067989dcc533ace2435?responseContentDisposition=attachment%3B%20filename%3DRepVGG_B3_200epochs.pdparams
[repvgg_b1g2]:https://bj.bcebos.com/v1/ai-studio-online/da4931eff12142a290ce8d01a0cd3b777a81b53c971b4dd2a1a627c615466570?responseContentDisposition=attachment%3B%20filename%3DRepVGG_B1g2.pdparams
[repvgg_b1g4]:https://bj.bcebos.com/v1/ai-studio-online/440040d200b14bcb9951e47877b7b416454affd75f8e4eaba6fedfa87c4ab66a?responseContentDisposition=attachment%3B%20filename%3DRepVGG_B1g4.pdparams
[repvgg_b2g4]:https://bj.bcebos.com/v1/ai-studio-online/42b0654c15f942c9828a7ca7d117638417c48ccdeac84123bcd72558db7a01c2?responseContentDisposition=attachment%3B%20filename%3DRepVGG_B2g4_200epochs.pdparams
[repvgg_b3g4]:https://bj.bcebos.com/v1/ai-studio-online/5e4f6084ee954a319c2e0c11aadae680c643ae88bdbb44d2a1875a38f5278060?responseContentDisposition=attachment%3B%20filename%3DRepVGG_B3g4_200epochs.pdparams

* Citation：

    ```
    @article{ding2021repvgg,
        title={RepVGG: Making VGG-style ConvNets Great Again},
        author={Ding, Xiaohan and Zhang, Xiangyu and Ma, Ningning and Han, Jungong and Ding, Guiguang and Sun, Jian},
        journal={arXiv preprint arXiv:2101.03697},
        year={2021}
    }
    ```
