# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 10:37:55 2021

@author: mlsol
"""
import torch as t
import torchlayers as tl
from complexPyTorch.complexLayers import (
    ComplexBatchNorm2d,
    ComplexConv2d,
    ComplexLinear,
)


def test_layers_shape_inference():
    #%% test shape inference of complex layers

    # complex linear layer
    MyComplexLinear = tl.infer(ComplexLinear)
    # Build and use just like any other layer in this library
    layer = tl.build(
        MyComplexLinear(out_features=32), t.randn(1, 64, dtype=t.complex64)
    )
    layer(t.randn(1, 64, dtype=t.complex64))
    print(layer)

    # PyTorch conv input is (N, C, H, W) format.
    # N is the number of samples/batch_size.
    # C is the channels.
    # H and W are height and width respectively.
    MyComplexConv2D = tl.infer(ComplexConv2d)

    # Build and use just like any other layer in this library
    layer = tl.build(
        MyComplexConv2D(out_channels=32), t.randn((1, 1, 64, 64), dtype=t.complex64)
    )
    layer(t.randn((1, 1, 64, 64), dtype=t.complex64))
    print(layer)

    MyComplexConv1D = tl.infer(ComplexConv2d)

    # Build and use just like any other layer in this library
    layer = tl.build(
        MyComplexConv2D(out_channels=32, kernel_size=(1, 3)),
        t.randn((1, 1, 1, 64), dtype=t.complex64),
    )
    layer(t.randn((1, 1, 1, 64), dtype=t.complex64))
    print(layer)

    MyComplexBatchNorm2d = tl.infer(ComplexBatchNorm2d)

    # Build and use just like any other layer in this library
    layer = tl.build(
        MyComplexBatchNorm2d(), t.randn([1, 32, 62, 62], dtype=t.complex64)
    )
    layer(t.randn([1, 32, 62, 62], dtype=t.complex64))
    print(layer)


def test_autograd_complex_grads():

    """
    dL/dz* = 0

    Example from https://pytorch.org/docs/stable/notes/autograd.html#complex-autograd-doc

    f(z) = cz

    where c is real

    f'(z) = c
    """
    import torch as t

    c = t.tensor([[1.0]], requires_grad=True)
    z = t.tensor([[1.0 + 1j]], requires_grad=True)

    f = c * z
    print(t.autograd.grad(f, z, create_graph=True))
    f.backward()
    print(z.grad)

    c = t.tensor([[1.0 + 1j]], requires_grad=True)
    z = t.tensor([[1.0 + 1j]], requires_grad=True)

    f = c * z
    print(t.autograd.grad(f, z, create_graph=True))
    f.backward()
    print(z.grad)

    #%%

    c = t.tensor([[1.0]], requires_grad=True)
    z = t.tensor([[1.0 + 1j]], requires_grad=True)
    f = c * z
    grads = t.autograd.grad(f, z, grad_outputs=-1j * f, create_graph=True)
    print(grads)

    #%%
    """
    https://github.com/pytorch/pytorch/issues/65711#issuecomment-971760607

    """
    import torch
    from torch.optim import (
        Adadelta,
        Adagrad,
        Adam,
        AdamW,
        SparseAdam,
        Adamax,
        ASGD,
        LBFGS,
        RMSprop,
        Rprop,
        SGD,
    )

    opti_list = [Adadelta, Adagrad, Adam, AdamW, ASGD, RMSprop, SGD]
    optimizer_constructor = lambda param: torch.optim.Adam([param], lr=0.001)

    for opt in opti_list:
        optimizer_constructor = lambda param: opt([param], lr=0.001)
        complex_param = torch.randn(1, 2, dtype=torch.complex64, requires_grad=True)
        complex_opt = optimizer_constructor(complex_param)
        real_param = torch.view_as_real(complex_param).detach().requires_grad_()
        real_opt = optimizer_constructor(real_param)

        for i in range(1):
            complex_param.grad = torch.randn(1, 2, dtype=torch.complex64)
            real_param.grad = torch.view_as_real(complex_param.grad)

            complex_opt.step()
            real_opt.step()
            print(torch.view_as_real(complex_param.grad) - real_param.grad)

    #%%

    import numpy as np
    import torch as t
    import complexPyTorch as ct
    from numpy import linalg

    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    data = t.tensor([[1 + 1j], [1 + 1j]], requires_grad=True).to(device)

    wa = [0.157 + 0.148j, 0.60 + 0.170j, 0.30 + 0.120j]  # (zs) the complex point zp[i]
    zp = [0.147 + 0.143j, 0.90 + 0.040j, 0.30 + 0.120j]  # (wa) will be in wa[i]

    # transformation parameters
    a = linalg.det(
        [
            [zp[0] * wa[0], wa[0], 1],
            [zp[1] * wa[1], wa[1], 1],
            [zp[2] * wa[2], wa[2], 1],
        ]
    )

    b = linalg.det(
        [
            [zp[0] * wa[0], zp[0], wa[0]],
            [zp[1] * wa[1], zp[1], wa[1]],
            [zp[2] * wa[2], zp[2], wa[2]],
        ]
    )

    c = linalg.det([[zp[0], wa[0], 1], [zp[1], wa[1], 1], [zp[2], wa[2], 1]])

    d = linalg.det(
        [
            [zp[0] * wa[0], zp[0], 1],
            [zp[1] * wa[1], zp[1], 1],
            [zp[2] * wa[2], zp[2], 1],
        ]
    )
    a = t.tensor(a, requires_grad=True)
    b = t.tensor(b, requires_grad=True)
    c = t.tensor(c, requires_grad=True)
    d = t.tensor(d, requires_grad=True)

    def mobius(z, a, b, c, d):
        return (a * z + b) / (c * z + d)

    L = mobius(z, a, b, c, d)

    L.sum().backward()
    print(data.grad)

    #%%
    import torch

    def mobius(z, a, b, c, d):
        if torch.isinf(z):
            if c == 0:
                return torch.inf
            return a / c
        if c * z + d == 0:
            return torch.inf
        else:
            return (a * z + b) / (c * z + d)

    def dmobius_dz(z, a, b, c, d):
        return (a * d - b * c) / (c * z + d) ** 2

    wa = [0.157 + 0.148j, 0.60 + 0.170j, 0.30 + 0.120j]  # (zs) the complex point zp[i]
    zp = [0.147 + 0.143j, 0.90 + 0.040j, 0.30 + 0.120j]  # (wa) will be in wa[i]

    # transformation parameters
    a = torch.linalg.det(
        torch.tensor(
            [
                [zp[0] * wa[0], wa[0], 1],
                [zp[1] * wa[1], wa[1], 1],
                [zp[2] * wa[2], wa[2], 1],
            ],
            requires_grad=False,
        )
    )

    b = torch.linalg.det(
        torch.tensor(
            [
                [zp[0] * wa[0], zp[0], wa[0]],
                [zp[1] * wa[1], zp[1], wa[1]],
                [zp[2] * wa[2], zp[2], wa[2]],
            ],
            requires_grad=False,
        )
    )

    c = torch.linalg.det(
        torch.tensor(
            [[zp[0], wa[0], 1], [zp[1], wa[1], 1], [zp[2], wa[2], 1]],
            requires_grad=False,
        )
    )

    d = torch.linalg.det(
        torch.tensor(
            [
                [zp[0] * wa[0], zp[0], 1],
                [zp[1] * wa[1], zp[1], 1],
                [zp[2] * wa[2], zp[2], 1],
            ],
            requires_grad=False,
        )
    )

    z = torch.tensor([[1.0 + 1j]], requires_grad=True)
    z_conj = torch.tensor([[1.0 - 1j]], requires_grad=True)

    dL_dz = dmobius_dz(z, a, b, c, d)
    dL_dz_conj = dmobius_dz(z_conj, a, b, c, d)

    L = mobius(z, a, b, c, d)
    L_conj = mobius(z_conj, a, b, c, d)

    L.backward()
    L_conj.backward()
    print(z_conj.grad == dL_dz)  # expectation: tensor([2.-2.j])
    print(dL_dz)
    #%% validating complex derivative in pytorch autograd
    import torch

    # https://complex-analysis.com/content/complex_differentiation.html
    z = torch.tensor([[1.0 + 1j]], requires_grad=True)
    f = z ** 2
    f.backward()
    print(z.grad)  # expectation: tensor([2.-2.j])

    # real y=xA+b
    z = torch.tensor([[1.0]], requires_grad=True)
    w = torch.tensor([[1.0]], requires_grad=True)
    b = torch.tensor([[1.0]], requires_grad=True)
    f = z @ w.t() + b
    f.backward()
    print(z.grad)  # expectation: tensor([[1.]])

    # complex y=xA+b
    z = torch.tensor([[1.0 + 1j]], requires_grad=True)
    w = torch.tensor([[1.0 + 1j]], requires_grad=True)
    b = torch.tensor([[1.0 + 1j]], requires_grad=True)
    f = z @ w.t() + b
    f.backward()
    print(z.grad)  # expectation: tensor([[1.-1.]])

    # complex matmul
    def complex_matmul(A, B):
        """
        Performs the matrix product between two complex matrices
        """

        outp_real = torch.matmul(A.real, B.real) - torch.matmul(A.imag, B.imag)
        outp_imag = torch.matmul(A.real, B.imag) + torch.matmul(A.imag, B.real)

        return outp_real.type(torch.complex64) + 1j * outp_imag.type(torch.complex64)

    z = torch.tensor([[1.0 + 1j]], requires_grad=True)
    w = torch.tensor([[1.0 + 1j]], requires_grad=True)
    b = torch.tensor([[1.0 + 1j]], requires_grad=True)
    f = complex_matmul(z, w.t()) + b
    f.backward()
    print(z.grad)  # expectation: tensor([[1.-1.]])

    import sys

    class myLinear(torch.nn.Module):
        def __init__(self, in_features, out_features, bias=True):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.bias = bias
            self.weight = torch.nn.Parameter(
                torch.randint(2, (out_features, in_features)).float()
            )
            self.bias = torch.nn.Parameter(torch.randint(2, (out_features,)).float())

        def forward(self, input):
            x, y = input.shape
            if y != self.in_features:
                sys.exit(
                    f"Wrong Input Features. Please use tensor with {self.in_features} Input Features"
                )
            output = input @ self.weight.t() + self.bias
            return output

    z = torch.tensor([[1.0]], requires_grad=True)
    w = torch.tensor([[1.0]], requires_grad=True)
    b = torch.tensor([[1.0]], requires_grad=True)
    f = myLinear(1, 1)(z)
    f.backward()
    print(z.grad)  # expectation: tensor([[1.-1.]])

    z = torch.tensor([[1.0]], requires_grad=True)
    w = torch.tensor([[1.0]], requires_grad=True)
    b = torch.tensor([[1.0]], requires_grad=True)
    f = myLinear(1, 1)(z)
    f.backward()
    print(z.grad)  # expectation: tensor([[1.-1.]])

    # %%
    dL_dz = t.autograd.grad(
        L,
        data.conj(),
        grad_outputs=data.data.new(data.shape).fill_(1 + 1j),
        create_graph=True,
    )

    print(dL_dz)

    # t.autograd.gradcheck(f, data)


if __name__ == "__main__":
    test_layers_shape_inference()
    test_autograd_complex_grads()
