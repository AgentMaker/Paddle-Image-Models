import ppim.models as models

from ppim.models import *
from inspect import isfunction, isclass

version = "1.1.0"
models_dict = {}
models_list = []


for k, v in models.__dict__.items():
    if isfunction(v):
        model_name = k.split("_")[0]
        if model_name not in models_dict:
            models_dict[model_name] = [k]
        else:
            models_dict[model_name].append(k)
    elif isclass(v):
        models_list.append(k)


def available():
    print("The pretrained models list as follow:")
    print("\n".join([str({k: v}) for k, v in models_dict.items()]))

    print("The models support as follow:")
    print("\n".join(str([item]) for item in models_list))
