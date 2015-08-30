def build_kwargs(dictionary, index):
    l = dictionary.get('length', 0)
    kwargs = dict()
    if index < l:
        keys = dictionary.keys()
        keys.remove('length')
        for key in keys:
            kwargs[key] = dictionary[key][index]

    return kwargs