def build_kwargs(dictionary, index):
    l = dictionary.get('length', 0)
    kwargs = dict()
    if index < l:
        keys = dictionary.keys()
        keys.remove('length')
        for key in keys:
            kwargs[key] = dictionary[key][index]

    return kwargs

def print_output_for_latex(output):
    for row in output:
        print '%i & %i & %.2f & %.2f & %.2f \\\\' % tuple(row)