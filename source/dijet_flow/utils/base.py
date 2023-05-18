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

#...preprocessing:

def logit(t, alpha=1e-6) -> torch.Tensor:
    x = alpha + (1 - 2 * alpha) * t
    return torch.log(x/(1-x))

def expit(t, alpha=1e-6) -> torch.Tensor:
    exp = torch.exp(t)
    x = exp / (1 + exp) 
    return (x - alpha) / (1 - 2*alpha)

#...kinematics:

def ep2ptepm(vec) -> torch.Tensor:
    """ Convert (px, py, pz, e) into (pT, eta, phi, mass)
        for torch.tensors.
    """
    px, py, pz, e = vec[:,0], vec[:,1], vec[:,2], vec[:,3]
    vec_ptepm = torch.zeros_like(vec)
    pT = torch.sqrt(px**2 + py**2) 
    p = torch.sqrt(px**2 + py**2 + pz**2)  # |p|
    cos = pz / p  # cos(theta)
    cos[p==0] = 1.0
    m2 = e**2 - p**2
    m2[m2 < 0] *= -1.0

    vec_ptepm[:,0] = torch.sqrt(px**2 + py**2)   # pT
    vec_ptepm[:,1][cos**2 < 1] = -0.5 * torch.log(1. - cos) + 0.5 * torch.log(1. + cos)   # eta
    vec_ptepm[:,1][(cos**2 >= 1) & (pz == 0.)] = 0.
    vec_ptepm[:,1][(cos**2 >= 1) & (pz > 0.)] = 10e10
    vec_ptepm[:,1][(cos**2 >= 1) & (pz < 0.)] = -10e10
    vec_ptepm[:,2] = torch.arctan2(py,px)     # phi
    vec_ptepm[:,2][(py == 0) & (px == 0)] = 0.0
    vec_ptepm[:,3] = torch.sqrt(m2)      # mass
    vec_ptepm[:,3][m2 < 0] *= -1.0
    return vec_ptepm

def em2ptepm(vec) -> torch.Tensor:
    """ Convert (px, py, pz, m) into (pT, eta, phi, mass)
        for torch.tensors.
    """
    vec_ptepm = torch.zeros_like(vec)
    px, py, pz, m = vec[:,0], vec[:,1], vec[:,2], vec[:,3]
    pT = torch.sqrt(px**2 + py**2)  # pT
    p = torch.sqrt(px**2 + py**2 + pz**2)  # |p|
    cos = pz / p  # cos(theta)
    cos[p==0] = 1.0

    vec_ptepm[:,0] = pT  
    vec_ptepm[:,1][cos**2 < 1] = -0.5 * torch.log(1. - cos) + 0.5 * torch.log(1. + cos)   # eta
    vec_ptepm[:,1][(cos**2 >= 1) & (pz == 0.)] = 0.
    vec_ptepm[:,1][(cos**2 >= 1) & (pz > 0.)] = 10e10
    vec_ptepm[:,1][(cos**2 >= 1) & (pz < 0.)] = -10e10
    vec_ptepm[:,2] = torch.arctan2(py,px)     # phi
    vec_ptepm[:,2][(py == 0) & (px == 0)] = 0.0
    vec_ptepm[:,3] = m      # mass
    return vec_ptepm

def ptepm2ep(vec) -> torch.Tensor:
    """ Convert (pT, eta, phi, mass) into (px, py, pz, e)
        for torch.tensors.
    """
    pt, eta, phi, m = vec[:,0], vec[:,1], vec[:,2], vec[:,3]
    vec_ep = torch.zeros_like(vec)
    vec_ep[:,0] = pt * torch.cos(phi)
    vec_ep[:,1] = pt * torch.sin(phi)
    vec_ep[:,2] = pt * torch.sinh(eta)
    vec_ep[:,3] = torch.sqrt(vec_ep[:,0]**2 + vec_ep[:,1]**2 + vec_ep[:,2]**2 + vec_ep[:,3]**2)
    return vec_ep

def ptepm2em(vec) -> torch.Tensor:
    """ Convert (pT, eta, phi, mass) into (px, py, pz, m)
        for torch.tensors.
    """
    pt, eta, phi, m = vec[:,0], vec[:,1], vec[:,2], vec[:,3]
    vec_em = torch.zeros_like(vec)
    vec_em[:,0] = pt * torch.cos(phi)
    vec_em[:,1] = pt * torch.sin(phi)
    vec_em[:,2] = pt * torch.sinh(eta)
    vec_em[:,3] = m
    return vec_em

def inv_mass(t1, t2, coord='ptepm') -> torch.Tensor:
    if coord == 'ptepm':
        t1 = ptepm2ep(t1)
        t2 = ptepm2ep(t2)
    px1, py1, pz1, e1 = t1[:,0], t1[:,1], t1[:,2], t1[:,3]
    px2, py2, pz2, e2 = t2[:,0], t2[:,1], t2[:,2], t2[:,3]
    m12 = torch.sqrt( (e1 + e2)**2 - (px1 + px2)**2 - (py1 + py2)**2 - (pz1 + pz2)**2 )
    return m12


def dphi(t1, t2, coord='ptepm') -> torch.Tensor:  
    if coord == 'ep':
        t1 = ep2ptepm(t1)
        t2 = ep2ptepm(t2)
    phi1, phi2 = t1[:,2], t2[:,2]
    del_phi = phi1 - phi2
    torch.pi = torch.acos(torch.zeros(1)).item() * 2
    del_phi[del_phi > torch.pi] -= 2 * torch.pi 
    del_phi[del_phi < -torch.pi] += 2 * torch.pi 
    return del_phi

def deta(t1, t2, coord='ptepm') -> torch.Tensor:  
    if coord == 'ep':
        t1 = ep2ptepm(t1)
        t2 = ep2ptepm(t2)
    eta1, eta2 = t1[:,1], t2[:,1]
    return eta1 - eta2

def deltaR(t1, t2, coord='ptepm') -> torch.Tensor:  
    del_phi = dphi(t1, t2, coord=coord)
    del_eta = deta(t1, t2, coord=coord)
    return torch.sqrt(del_eta**2 + del_phi**2)

