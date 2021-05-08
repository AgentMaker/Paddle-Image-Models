from paddle.utils.download import get_weights_path_from_url


def load_model(model, url):
    path = get_weights_path_from_url(url)
    model.set_state_dict(paddle.load(path))
    return model
