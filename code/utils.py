import pandas as pd


def value_dist(dict_data):
    dict_res = {}
    for name in dict_data:
        data = dict_data[name].value_counts()
        n = data.sum()
        data_counts = {
            key: f"{value} ({round(value/n*100)}%)"
            for key, value in data.to_dict().items()
        }
        dict_res[name] = data_counts
    return pd.DataFrame(dict_res)
