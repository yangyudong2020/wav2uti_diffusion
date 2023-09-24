import math
import copy
from random import random
from functools import partial, wraps
import torch
from torch import nn, einsum

# helper functions

def exists(val):
    return val is not None

def identity(t, *args, **kwargs):
    return t

def divisible_by(numer, denom):
    return (numer % denom) == 0

def first(arr, d = None):
    if len(arr) == 0:
        return d
    return arr[0]
    
def find_first(fn, arr):
    for ind, el in enumerate(arr):
        if fn(el):
            return ind
    return -1

def maybe(fn):
    @wraps(fn)
    def inner(x):
        if not exists(x):
            return x
        return fn(x)
    return inner

def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cast_tuple(val, length = None):
    if isinstance(val, list):
        val = tuple(val)

    output = val if isinstance(val, tuple) else ((val,) * default(length, 1))

    if exists(length):
        assert len(output) == length

    return output

def compact(input_dict):
    return {key: value for key, value in input_dict.items() if exists(value)}

def maybe_transform_dict_key(input_dict, key, fn):
    if key not in input_dict:
        return input_dict

    copied_dict = input_dict.copy()
    copied_dict[key] = fn(copied_dict[key])
    return copied_dict

def cast_uint8_images_to_float(images):
    if not images.dtype == torch.uint8:
        return images
    return images / 255
    
def zero_init_(m):
    nn.init.zeros_(m.weight)
    if exists(m.bias):
        nn.init.zeros_(m.bias)
        
def pad_tuple_to_length(t, length, fillvalue = None):
    remain_length = length - len(t)
    if remain_length <= 0:
        return t
    return (*t, *((fillvalue,) * remain_length))
    
def pick_and_pop(keys, d):
    values = list(map(lambda key: d.pop(key), keys))
    return dict(zip(keys, values))

def group_dict_by_key(cond, d):
    return_val = [dict(),dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)

def string_begins_with(prefix, str):
    return str.startswith(prefix)

def group_by_key_prefix(prefix, d):
    return group_dict_by_key(partial(string_begins_with, prefix), d)

def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(partial(string_begins_with, prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr
    
    
# url to fs, bucket, path - for checkpointing to cloud

def url_to_bucket(url):
    if '://' not in url:
        return url

    _, suffix = url.split('://')

    if prefix in {'gs', 's3'}:
        return suffix.split('/')[0]
    else:
        raise ValueError(f'storage type prefix "{prefix}" is not supported yet')
        
        
        
        
class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

def Sequential(*modules):
    return nn.Sequential(*filter(exists, modules))