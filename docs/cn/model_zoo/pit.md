# PiT
* 论文：[Rethinking Spatial Dimensions of Vision Transformers](https://arxiv.org/abs/2103.16302)
* 官方项目：[naver-ai/pit](https://github.com/naver-ai/pit)
* 模型代码：[pit.py](../../../ppim/models/pit.py)
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
    ```

* 模型细节：

    |         Model         |     Model Name        | Params (M) | FLOPs (G) | Top-1 (%) | Top-5 (%) |      Pretrained Model        |
    |:---------------------:|:---------------------:|:----------:|:---------:|:---------:|:---------:|:----------------------------:|
    | PiT-Ti                | pit_ti                | 4.9        | 0.7       | 72.91     |  91.40    | [Download][pit_ti]           |
    | PiT-XS                | pit_xs                | 10.6       | 1.4       | 78.18     |  94.16    | [Download][pit_xs]           |
    | PiT-S                 | pit_s                 | 23.5       | 2.9       | 81.08     |  95.33    | [Download][pit_s]            |
    | PiT-B                 | pit_b                 | 73.8       | 12.5      | 82.44     |  95.71    | [Download][pit_b]            |
    | PiT-Ti distilled      | pit_ti_distilled      | 4.9        | 0.7       | 74.54     |  92.10    | [Download][pit_ti_distilled] |
    | PiT-XS distilled      | pit_xs_distilled      | 10.6       | 1.4       | 79.31     |  94.36    | [Download][pit_xs_distilled] |
    | PiT-S distilled       | pit_s_distilled       | 23.5       | 2.9       | 81.99     |  95.79    | [Download][pit_s_distilled]  |
    | PiT-B distilled       | pit_b_distilled       | 73.8       | 12.5      | 84.14     |  96.86    | [Download][pit_b_distilled]  |


[pit_ti]:https://bj.bcebos.com/v1/ai-studio-online/3d0fe9a33bb74abaa0648f6200b37e5b49ca9a4f15a04afbab7a885da64dfa62?responseContentDisposition=attachment%3B%20filename%3Dpit_ti.pdparams
[pit_xs]:https://bj.bcebos.com/v1/ai-studio-online/4bee539cc81a477a8bae4795f91d583c810ea4832e6d4ed983b37883669e6a6d?responseContentDisposition=attachment%3B%20filename%3Dpit_xs.pdparams
[pit_s]:https://bj.bcebos.com/v1/ai-studio-online/232c216331d04fb58f77839673b34652ea229a9ab84044a493e08cd802ab4fe3?responseContentDisposition=attachment%3B%20filename%3Dpit_s.pdparams
[pit_b]:https://bj.bcebos.com/v1/ai-studio-online/26f33b44d9424626b74eb7cfad2041582afabdebd6474afa976cc0a55c226791?responseContentDisposition=attachment%3B%20filename%3Dpit_b.pdparams
[pit_ti_distilled]:https://bj.bcebos.com/v1/ai-studio-online/9707c73717274b5e880e8401b85dcf9ad12b0d7e47944af68b3d6a2236b70567?responseContentDisposition=attachment%3B%20filename%3Dpit_ti_distill.pdparams
[pit_xs_distilled]:https://bj.bcebos.com/v1/ai-studio-online/61aa3339366d4315854bf67a8df1cea20f4a2402b2d94d7688d995423a197df1?responseContentDisposition=attachment%3B%20filename%3Dpit_xs_distill.pdparams
[pit_s_distilled]:https://bj.bcebos.com/v1/ai-studio-online/65acbfa1d6a94c689225fe95c6ec48567f5c05ee051243d6abe3bbcbd6119f5d?responseContentDisposition=attachment%3B%20filename%3Dpit_s_distill.pdparams
[pit_b_distilled]:https://bj.bcebos.com/v1/ai-studio-online/2d6631b21542486b8333440c612847f35a7782d2890f4514ad8007c34ae77e66?responseContentDisposition=attachment%3B%20filename%3Dpit_b_distill.pdparams


* 引用：

    ```
    @article{heo2021pit,
        title={Rethinking Spatial Dimensions of Vision Transformers},
        author={Byeongho Heo and Sangdoo Yun and Dongyoon Han and Sanghyuk Chun and Junsuk Choe and Seong Joon Oh},
        journal={arXiv: 2103.16302},
        year={2021},
    }
    ```
