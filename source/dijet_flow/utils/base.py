import torch
import numpy as np
import os.path
import shutil
import itertools
import json
import inspect
import argparse

def make_dir(path, overwrite=False, sub_dirs=False, verbose=True):  
    Directory = path
    if overwrite:
        shutil.rmtree(Directory, ignore_errors=True)
        os.mkdir(Directory)
    else:
        for I in itertools.count():
            Directory = path + '__' + str(I+1)
            if not os.path.isdir(Directory):
                os.mkdir(Directory)
                break
            else:
                continue
    if sub_dirs:
        for d in sub_dirs: 
            os.mkdir(Directory+'/'+d)
    if verbose:
        print("#================================================")
        print("INFO: working directory: {}".format(Directory))
        print("#================================================")
    return Directory


def savefig(filename, extension="png"):
    counter = 1
    base_filename, ext = os.path.splitext(filename)
    if ext == "":
        ext = f".{extension}"
    unique_filename = f"{base_filename}{ext}"
    while os.path.exists(unique_filename):
        unique_filename = f"{base_filename}_{counter}{ext}"
        counter += 1
    return unique_filename

def copy_parser(original_parser, description, modifications=False):
    new_parser = argparse.ArgumentParser(description=description)
    for action in original_parser._actions:
        if action.dest == 'help':
            continue
        kwargs = {'dest':action.dest, 'type':action.type, 'help':action.help, 'default':action.default, 'required':action.required}
        if modifications:
            if action.dest in modifications:
                kwargs.update(modifications[action.dest])
        new_parser.add_argument(action.option_strings[0], **kwargs)
    return new_parser


def serialize(obj, name_only=True):
    if callable(obj):
        if name_only:
            return obj.__name__
        else:
            return inspect.getsource(obj).strip()
    return obj

def save_arguments(args, name):
    args_dict = {a: serialize(b) for a, b in vars(args).items()}
    with open(args.workdir+'/'+name, 'w') as file: json.dump(args_dict, file, indent=4)


def get_gpu_memory():
    torch.cuda.empty_cache()
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    reserved = torch.cuda.memory_reserved(0) / (1024**3)
    allocated = torch.cuda.memory_allocated(0) / (1024**3)
    free = total - reserved
    return {
        'total': total,
        'reserved': reserved,
        'allocated': allocated,
        'free': free
    }

def shuffle(tensor) -> torch.Tensor:
    indices = torch.randperm(tensor.size(0))
    return tensor[indices]