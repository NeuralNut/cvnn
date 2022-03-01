# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 13:36:13 2022

@author: mlsol
"""
import torch as t
from torch.nn.functional import leaky_relu

def euler(z):
    a, b = z.real, z.imag
    r = t.sqrt(a**2 + b**2)
    theta = t.atan(b/a)
    theta[a < 0] += t.pi
    return r, theta


def magnitude(z):
    return t.sqrt(z.real**2 + z.imag**2)

def angle(z):
    a, b = z.real, z.imag
    theta = t.atan(b / a)
    theta[a < 0] += t.pi
    theta = t.fmod(theta, 2*t.pi)
    return theta

def phase(z):
    a, b = z.real, z.imag
    theta = t.atan(b / a)
    theta[a < 0] += t.pi
    return theta
    

def zrelu(z):
    '''
    Guberman ReLU:
    Nitzan Guberman. On complex valued convolutional neural networks. arXiv preprint arXiv:1602.09046, 2016
    Eq.(5)
    https://arxiv.org/pdf/1705.09792.pdf
    '''
    a, b = z.real, z.imag
    mask = ((0 < angle(z)) * (angle(z) < t.pi/2)).float()
    real = a * mask
    imag = b * mask
    
    return real + imag*1j

def modrelu(z, bias):
    '''
    Martin Arjovsky, Amar Shah, and Yoshua Bengio. Unitary evolution recurrent neural networks. arXiv preprint arXiv:1511.06464, 2015.
    Notice that |z| (z.magnitude) is always positive, so if b > 0  then |z| + b > = 0 always.
    In order to have any non-linearity effect, b must be smaller than 0 (b<0).
    '''
    a, b = z.real, z.imag
    z_mag = magnitude(z)
    mask = ((z_mag + bias) >= 0).float() * (1 + bias / z_mag)
    real = mask * a
    imag = mask * b
    return real + imag*1j

def complex_leaky_relu(z):
    return leaky_relu(z.real)+leaky_relu(z.imag)*1j