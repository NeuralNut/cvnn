# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 16:40:57 2022

@author: mlsol
"""
import torch as t
import torchlayers as tl
import complexPyTorch as ct
from complexPyTorch.complexLayers import ComplexBatchNorm2d, ComplexBatchNorm1d, ComplexConv2d
from complexPyTorch.complexLayers import ComplexLinear, ComplexReLU
from complexPyTorch.complexLayers import ComplexDropout, ComplexDropout2d
from complexPyTorch.complexLayers import ComplexMaxPool2d


from torch.nn import AdaptiveAvgPool2d
from torch.nn.functional import adaptive_avg_pool2d


Linear = tl.infer(t.nn.Linear)
ComplexLinear = tl.infer(ComplexLinear)
ComplexConv2D = tl.infer(ComplexConv2d)
ComplexBatchNorm2D = tl.infer(ComplexBatchNorm2d)
ComplexBatchNorm1D = tl.infer(ComplexBatchNorm1d)
ComplexReLU = tl.infer(ComplexReLU)
ComplexDropout = ComplexDropout
ComplexDropout2D = ComplexDropout2d
# ComplexMaxPool2D = tl.infer(ComplexMaxPool2d)
ComplexMaxPool2D = ComplexMaxPool2d


def complex_adap_avg_pool2d(input, *args, **kwargs):
    '''
    Perform complex average pooling.
    '''    
    absolute_value_real = adaptive_avg_pool2d(input.real, *args, **kwargs)
    absolute_value_imag =  adaptive_avg_pool2d(input.imag, *args, **kwargs)    
    
    return absolute_value_real.type(t.complex64)+1j*absolute_value_imag.type(t.complex64)

class ComplexAdapAvgPool2D(t.nn.Module):

    def __init__(self, output_size):
        super(ComplexAdapAvgPool2D,self).__init__()
        self.output_size = output_size
    
    def forward(self,input):
        return complex_adap_avg_pool2d(input, 
                                       output_size=self.output_size
                                       )

class ComplexInverseDropout(t.nn.Module):
    def __init__(self, p: float = 0.5):
        super(ComplexInverseDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
        self.p = p

    def forward(self, x):
        if self.training:
            binomial = t.distributions.binomial.Binomial(probs=1-self.p)
            mask = (binomial.sample(x.size()) * (1.0/(1-self.p))).to('cuda')
            return x * mask
        return x


# Complex to Real Transforms
class Abs(t.nn.Module):
    """ Custom absolute value layer """
    def __init__(self):
        super(Abs, self).__init__()
    def forward(self, x):
        x = t.abs(x)
        return x

class Magnitude(t.nn.Module):
    '''
    If z = u+vi, Magnitude(z) = (conj(z)*z)^.5 = (u^2 + v^2)^.5
    Nomenclature from Monning and Manandhar 2018 https://arxiv.org/pdf/1811.12351.pdf
    '''
    def __init__(self):
        super(Magnitude, self).__init__()

    def forward(self, x):
        x = t.conj(x) * x
        x = t.sqrt(x)
        x = x.type(t.float32)
        return x


class Intensity(t.nn.Module):
    '''
    If z = u+vi, Intensity(z) = conj(z)*z = u^2 + v^2
    Nomenclature from Monning and Manandhar 2018 https://arxiv.org/pdf/1811.12351.pdf
    '''
    def __init__(self):
        super(Intensity, self).__init__()

    def forward(self, x):
        x = t.conj(x) * x
        x = x.type(t.float32)
        return x

class Conj(t.nn.Module):
    '''
    Conjugate of a complex number
    '''
    def __init__(self):
        super(Conj, self).__init__()

    def forward(self, x):
        return t.conj(x) 
    
    
class GlobalAveragePooling2D(t.nn.Module):
    '''
    Average over spatial dimensions (N,C,H,W) --> (N,C)
    '''
    def __init__(self):
        super(GlobalAveragePooling2D, self).__init__()

    def forward(self, x):
        return x.mean([-1,-2])
    
    
class GlobalAveragePooling1D(t.nn.Module):
    '''
    Average over spatial dimensions (N,C,L) --> (N,C)
    '''
    def __init__(self):
        super(GlobalAveragePooling1D, self).__init__()

    def forward(self, x):
        return x.mean([-1])