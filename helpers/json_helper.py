# coding=utf-8

import json


def save_json(obj, path):
    with open(path, 'w') as fp:
        json.dump(obj, fp, indent=True)


def load_json(path):
    with open(path) as fp:
        return json.load(fp)
