import copy

def override_config(configs, override_list):
    new_configs = copy.deepcopy(configs)
    for item in override_list:
        arr = item.split()
        if len(arr) != 2:
            print(f"the overrive {item} format not correct, skip it")
            continue
        keys = arr[0].split('.')
        s_configs = new_configs
        for i, key in enumerate(keys):
            if key not in s_configs:
                print(f"the overrive {item} format not correct, skip it")
            if i == len(keys) - 1:
                param_type = type(s_configs[key])
                if param_type != bool:
                    s_configs[key] = param_type(arr[1])
                else:
                    s_configs[key] = arr[1] in ['true', 'True']
                print(f"override {arr[0]} with {arr[1]}")
            else:
                s_configs = s_configs[key]
    return new_configs
