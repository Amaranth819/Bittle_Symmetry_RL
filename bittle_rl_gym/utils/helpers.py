import yaml
import os


def class_to_dict(obj) -> dict:
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result



def write_dict_to_yaml(dict_data, yaml_file_name = 'config.yaml'):    
    with open(yaml_file_name, 'w') as f:
        yaml.dump(dict_data, f)