from collections import defaultdict

list_of_dict = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
def listdict_to_dictlist(list_of_dict):
    out = defaultdict(list)
    for d in list_of_dict:
        for k, v in d.items():
            out[k].append(v)
    return dict(out)

print(listdict_to_dictlist(list_of_dict))