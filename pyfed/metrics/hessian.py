# *
# @file Different utility functions
# Copyright (c) Zhewei Yao, Amir Gholami
# All rights reserved.
# This file is part of PyHessian library.
#
# PyHessian is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyHessian is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyHessian.  If not, see <http://www.gnu.org/licenses/>.
# *

import numpy as np
import torch
from torch import autograd
from torch.utils.data import DataLoader


class Hessian:
    """
    The class used to compute :
        i) the top 1 (n) eigenvalue(s) of the neural network
        ii) the trace of the entire neural network
        iii) the estimated eigenvalue density
    """

    def __init__(self, model, criterion, dataset, batch_size=32, full=False, device='cuda'):
        """
        model: the model that needs Hessian information
        criterion: the loss function
        dataset: pytorch Dataset compatible object, including inputs and its corresponding labels
        batch_size: the batch size used for computation. Only used when n_samples is -1
        full: if True, use all the data, otherwise, use a single batched data
        device: the device used for computation
        """
        self.model = model.eval()  # make model is in evaluation mode
        self.criterion = criterion
        self.device = device
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        self.full = full

        # pre-compute for single batch case to simplify the computation.
        if not full:
            self.inputs, self.targets = next(iter(self.dataloader))
            self.inputs, self.targets = self.inputs.to(self.device), self.targets.to(self.device)
            # if we only compute the Hessian information for a single batch data, we can re-use the gradients.
            outputs = self.model(self.inputs)
            loss = self.criterion(outputs, self.targets)
            # loss.backward(create_graph=True)
            self.gradsH = autograd.grad(loss, self.model.parameters(), create_graph=True)
            self.params = list(self.model.parameters())

    def __repr__(self):
        return str(
            'Hessian Calculator\n'
            f'\t{"Model":^11} : {self.model.__class__.__name__}\n'
            f'\t{"Criterion":^11} : {str(self.criterion)}\n'
            f'\t{"Dataset":^11} : {self.dataset.__class__.__name__}\n'
            f'\t{"Device":^11} : {self.device}'
        )

    def __str__(self):
        return self.__repr__()

    def dataloader_hv_product(self, v):
        """
        Compute the Hessian-vector product for a hessian matrix of loss from total data and a given vector
        """
        num_data = 0  # count the number of datum points in the dataloader
        THv = [torch.zeros(p.size()).to(self.device) for p in self.params]  # accumulate result
        for inputs, targets in self.dataset:
            self.model.zero_grad()
            tmp_num_data = inputs.size(0)
            outputs = self.model(inputs.to(self.device))
            loss = self.criterion(outputs, targets.to(self.device))
            loss.backward(create_graph=True)
            params, gradsH = get_params_grad(self.model)
            self.model.zero_grad()
            Hv = torch.autograd.grad(gradsH, params, grad_outputs=v, only_inputs=True, retain_graph=False)
            THv = [THv1 + Hv1 * float(tmp_num_data) + 0. for THv1, Hv1 in zip(THv, Hv)]
            num_data += float(tmp_num_data)

        THv = [THv1 / float(num_data) for THv1 in THv]
        eigenvalue = group_product(THv, v).cpu().item()
        return eigenvalue, THv

    def eigenvalues(self, max_iter=100, tol=1e-3, top_n=1, seed=42, create_graph=False):
        """
        compute the top_n eigenvalues using power iteration method
        maxIter: maximum iterations used to compute each single eigenvalue
        tol: the relative tolerance between two consecutive eigenvalue computations from power iteration
        top_n: top top_n eigenvalues will be computed
        """
        assert top_n >= 1

        eigenvalues = []
        eigenvectors = []

        computed_dim = 0

        torch.manual_seed(seed)
        while computed_dim < top_n:
            eigenvalue = None
            # generate random vector
            v = [torch.randn(p.size()).to(self.device) for p in self.params]
            # normalize
            v = normalize(v)

            for i in range(max_iter):
                v = orthnormal(v, eigenvectors)
                self.model.zero_grad()

                if self.full:
                    tmp_eigenvalue, Hv = self.dataloader_hv_product(v)
                else:
                    Hv = hessian_vector_product(self.gradsH, self.params, v, create_graph=create_graph)
                    tmp_eigenvalue = group_product(Hv, v).cpu().item()

                v = normalize(Hv)

                if eigenvalue is None:
                    eigenvalue = tmp_eigenvalue
                else:
                    if abs(eigenvalue - tmp_eigenvalue) / (abs(eigenvalue) + 1e-6) < tol:
                        break
                    else:
                        eigenvalue = tmp_eigenvalue
            eigenvalues.append(eigenvalue)
            eigenvectors.append(v)
            computed_dim += 1

        return eigenvalues, eigenvectors

    def trace(self, max_iter=100, tol=1e-3, seed=42):
        """
        compute the trace of hessian using Hutchinson's method
        maxIter: maximum iterations used to compute trace
        tol: the relative tolerance
        """

        device = self.device
        trace_vhv = []
        trace = 0.
        torch.manual_seed(seed)
        for i in range(max_iter):
            self.model.zero_grad()
            v = [
                torch.randint_like(p, high=2, device=device)
                for p in self.params
            ]
            # Generate Rademacher random variables
            for v_i in v:
                v_i[v_i == 0] = -1

            if self.full:
                _, Hv = self.dataloader_hv_product(v)
            else:
                Hv = hessian_vector_product(self.gradsH, self.params, v)
            trace_vhv.append(group_product(Hv, v).cpu().item())
            if abs(np.mean(trace_vhv) - trace) / (abs(trace) + 1e-6) < tol:
                return trace_vhv
            else:
                trace = np.mean(trace_vhv)

        return trace_vhv

    def density(self, n_iter=100, n_slq=1, seed=42):
        """
        compute estimated eigenvalue density using stochastic lanczos algorithm (SLQ)
        iter: number of iterations used to compute trace
        n_v: number of SLQ runs
        """
        device = self.device
        eigen_list_full = []
        weight_list_full = []

        torch.manual_seed(seed)
        for k in range(n_slq):
            v = [torch.randint_like(p, high=2, device=device) for p in self.params]
            # Generate Rademacher random variables
            for v_i in v:
                v_i[v_i == 0] = -1
            v = normalize(v)

            # Standard lanczos algorithm initlization
            v_list = [v]
            w_list = []
            alpha_list = []
            beta_list = []
            # Lanczos iteration
            for i in range(n_iter):
                self.model.zero_grad()
                w_prime = [torch.zeros(p.size()).to(device) for p in self.params]
                if i == 0:
                    if self.full:
                        _, w_prime = self.dataloader_hv_product(v)
                    else:
                        w_prime = hessian_vector_product(self.gradsH, self.params, v)
                    alpha = group_product(w_prime, v)
                    alpha_list.append(alpha.cpu().item())
                    w = group_add(w_prime, v, alpha=-alpha)
                    w_list.append(w)
                else:
                    beta = torch.sqrt(group_product(w, w))
                    beta_list.append(beta.cpu().item())
                    if beta_list[-1] != 0.:
                        # We should re-orth it
                        v = orthnormal(w, v_list)
                        v_list.append(v)
                    else:
                        # generate a new vector
                        w = [torch.randn(p.size()).to(device) for p in self.params]
                        v = orthnormal(w, v_list)
                        v_list.append(v)
                    if self.full:
                        _, w_prime = self.dataloader_hv_product(v)
                    else:
                        w_prime = hessian_vector_product(
                            self.gradsH, self.params, v)
                    alpha = group_product(w_prime, v)
                    alpha_list.append(alpha.cpu().item())
                    w_tmp = group_add(w_prime, v, alpha=-alpha)
                    w = group_add(w_tmp, v_list[-2], alpha=-beta)

            A = torch.zeros(n_iter, n_iter).to(device)
            for i in range(len(alpha_list)):
                A[i, i] = alpha_list[i]
                if i < len(alpha_list) - 1:
                    A[i + 1, i] = beta_list[i]
                    A[i, i + 1] = beta_list[i]
            L, V = torch.linalg.eig(A)

            eigen_list = L[:, 0]
            weight_list = V[0, :] ** 2
            eigen_list_full.append(list(eigen_list.cpu().numpy()))
            weight_list_full.append(list(weight_list.cpu().numpy()))

        return eigen_list_full, weight_list_full


def group_product(xs, ys):
    """
    the inner product of two lists of variables xs,ys
    :param xs:
    :param ys:
    :return:
    """
    return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])


def group_add(params, update, alpha=1):
    """
    params = params + update*alpha
    :param params: list of variable
    :param update: list of data
    :return:
    """
    for i, p in enumerate(params):
        params[i].data.add_(update[i] * alpha)
    return params


def normalize(ts, eps=1e-12):
    """
    normalization of a list of tensors
    view list of tensors as a single concatenated and flattened vector as
    :param ts: list of tensors
    return: normalized vectors v
    """
    s = torch.sqrt(sum([torch.sum(t ** 2) for t in ts])) + 1e-12
    ts = [t / (s + eps) for t in ts]
    return ts


def get_params_grad(model):
    """
    get model parameters and corresponding gradients
    """
    params = []
    grads = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        params.append(param)
        grads.append(0. if param.grad is None else param.grad + 0.)
    return params, grads


def hessian_vector_product(gradsH, params, v, create_graph=False):
    """
    compute the hessian vector product certain point and vector
    :param gradsH: the gradient at the current point,
    :param params: the corresponding variables,
    :param v: the vector to be multiplied with the hessian,
    :param create_graph: whether to create a new graph. set to True to compute the third order derivative.
    """
    # only_inputs argument is set to True by default and now deprecated
    return torch.autograd.grad(
        gradsH, params, grad_outputs=v,
        retain_graph=True, create_graph=create_graph
    )


def orthnormal(w, v_list):
    """
    make vector w orthogonal to each vector in v_list.
    afterwards, normalize the output w
    """
    for v in v_list:
        w = group_add(w, v, alpha=-group_product(w, v))
    return normalize(w)
