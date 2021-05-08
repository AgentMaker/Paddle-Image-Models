# CoaT
* Paper：[Co-Scale Conv-Attentional Image Transformers](https://arxiv.org/abs/2104.06399)
* Origin Repo：[mlpc-ucsd/CoaT](https://github.com/mlpc-ucsd/CoaT)
* Code：[coat.py](../../../ppim/models/coat.py)

* Model Details：

    |         Model           |       Model Name        | Params (M) | FLOPs (G) | Top-1 (%) | Top-5 (%) |          Pretrained Model        |
    |:-----------------------:|:-----------------------:|:----------:|:---------:|:---------:|:---------:|:--------------------------------:|
    | CoaT-tiny               |  coat_ti                |  5.5       |  4.4      | 72.18     |  91.11    | [Download][coat_ti]              |
    | CoaT-mini               |  coat_m                 | 10.0       |  6.8      | 81.99     |  95.74    | [Download][coat_lite_m]          |
    | CoaT-lite-tiny          |  coat_lite_ti           |  5.7       |  1.6      | 79.85     |  95.04    | [Download][coat_lite_ti]         |
    | CoaT-lite-mini          |  coat_lite_m            | 11.0       |  2.0      | 81.99     |  95.74    | [Download][coat_lite_m]          |


[coat_ti]https://bj.bcebos.com/v1/ai-studio-online/5250fdb938de4126a25f9d3f84b75ab114a268349b8744afb159408b6797ca81?responseContentDisposition=attachment%3B%20filename%3Dcoat_tiny.pdparams
[coat_lite_ti]https://bj.bcebos.com/v1/ai-studio-online/e33788c2a6e540b3aa92b169ed0ea2c61eff43479ff644d98cdb767f33bcc199?responseContentDisposition=attachment%3B%20filename%3Dcoat_lite_tiny.pdparams
[coat_lite_m]https://bj.bcebos.com/v1/ai-studio-online/c303c26af4974cfb97bd9b9dc400a4d5981c43fc149a401e937cd0186f31b92c?responseContentDisposition=attachment%3B%20filename%3Dcoat_lite_mini.pdparams

* Citation：

    ```
    @misc{xu2021coscale,
        title={Co-Scale Conv-Attentional Image Transformers}, 
        author={Weijian Xu and Yifan Xu and Tyler Chang and Zhuowen Tu},
        year={2021},
        eprint={2104.06399},
        archivePrefix={arXiv},
        primaryClass={cs.CV}
    }
    ```
