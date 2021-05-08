import paddle


def load_model(model, url):
    path = paddle.utils.download.get_weights_path_from_url(url)
    model.set_state_dict(paddle.load(path))
    return model
