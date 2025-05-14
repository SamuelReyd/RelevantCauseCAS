def add_dict_value(d, value, keys):
    if len(keys) == 1:
        if keys[0] not in d:
            d[keys[0]] = []
        d[keys[0]].append(value)
    else:
        if keys[0] not in d:
            d[keys[0]] = {}
        add_dict_value(d[keys[0]], value, keys[1:])

def set_dict_value(d, value, keys):
    if len(keys) == 1:
        d[keys[0]] = value
    else:
        if keys[0] not in d:
            d[keys[0]] = {}
        set_dict_value(d[keys[0]], value, keys[1:])


def get_3_axes(**kwargs):
    _, axes = plt.subplots(1,3, figsize=(9,3), **kwargs)
    return axes