import torch
import numpy as np
import os.path
import shutil
import itertools
import json
import inspect
import argparse

def make_dir(path, overwrite=False, sub_dirs=False):  
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


def ep2ptepm(vec):
    """ Convert (px, py, pz, e) into (pT, eta, phi, mass)
        for torch.tensors.
    """
    vec_ = torch.zeros_like(vec)
    p = torch.sqrt(vec[:,0]**2 + vec[:,1]**2 + vec[:,2]**2)  # |p|
    cos = vec[:,3] / p  # cos(theta)
    cos[p==0] = 1.0
    cos_0 = cos**2 < 1
    m2 = vec[:,3]**2 - p**2
    m2[m2 < 0] *= -1.0
    vec_[:,0] = torch.sqrt(vec[:,0]**2 + vec[:,1]**2)   # pT
    vec_[:,1][cos_0] = -0.5 * torch.log(1. - cos) + 0.5 * torch.log(1. + cos)   # eta
    vec_[:,1][~cos_0 & (vec[:,2] == 0.)] = 0.
    vec_[:,1][~cos_0 & (vec[:,2] > 0.)] = 10e10
    vec_[:,1][~cos_0 & (vec[:,2] < 0.)] = -10e10
    vec_[:,2] = torch.arctan2(vec[:,1], vec[:,0])     # phi
    vec_[:,2][(vec[:,1] == 0) & (vec[:,0] == 0)] = 0.
    vec_[:,3] = np.sqrt(m2)      # mass
    vec_[:,3][m2 < 0] *= -1.0
    return vec_