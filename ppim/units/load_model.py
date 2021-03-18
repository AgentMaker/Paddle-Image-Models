import os
import wget
import paddle


def load_model(model, url):
    file_name = url.split(r'filename%3D')[-1]
    model_path = os.path.join('pretrained_models', file_name)
    if not os.path.isfile(model_path):
        if not os.path.exists('pretrained_models'):
            os.mkdir('pretrained_models')
        wget.download(url, out=model_path)
    params = paddle.load(model_path)
    model.set_dict(params)
    return model
