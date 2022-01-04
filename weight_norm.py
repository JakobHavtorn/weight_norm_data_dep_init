"""
Module for weight normalization with data dependent initialization.

The output after weight normalization is computed as

    y = w * x + b ,  where   w = g * v / ||v|| ,
        = g * v / ||v|| * x + b

where the weight parameter w is reparameterized by a scale parameter g
and normed weight v. We refer to w as the primary weight.

The data dependent normalization can be derived as follows:

We can write the expected value of the output as

    E[y] = E[g * v / ||v|| * x - b]
            = g * E[v / ||v|| * x] - b

If we set b = g * E[v / ||v|| * x] + µ, then

    E[y] = g * E[v / ||v|| * x] - g * E[v / ||v|| * x] + µ
    E[y] = µ

for a certain mean of choice µ.

Likewise, for the variance, we can write

    V[y] = V[g * v / ||v|| * x - b]
    V[y] = g^2 * V[v / ||v|| * x] - V[b]
    V[y] = g^2 * V[v / ||v|| * x]

If we then set g = σ / sqrt(V[v / ||v|| * x]) then

    V[y] = σ^2 / V[v / ||v|| * x] * V[v / ||v|| * x]
    V[y] = σ^2

for a certain standard deviation of choice σ.

In terms of the standard deviation, sqrt(V[y]) = σ.

We can compute v / ||v|| * x by a forward pass through the weight
normalized module where we set the bias b = 0 and the scale g = 1
and then compte V[v / ||v|| * x] and E[v / ||v|| * x] as statistics
over the batch.
"""


from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import uuid

import torch.nn as nn


REMOVER_HOOK_HANDLES = dict()


def hook_remover_hook(
    module: nn.Module,
    input: torch.Tensor,
    output: torch.Tensor,
    handles: List[torch.utils.hooks.RemovableHandle],
    own_hook_id: str,
):
    """A forward hook that when called removes the other given hooks (`handles`) and then removes itself.

    Intended to be used as a forward hook. It will remove itself after executing and hence will be called only once.
    """
    if own_hook_id not in REMOVER_HOOK_HANDLES:
        raise RuntimeError(f"Remover hook {own_hook_id=} not found in index:\n{REMOVER_HOOK_HANDLES=}")

    for handle in handles:
        handle.remove()

    REMOVER_HOOK_HANDLES[own_hook_id].remove()  # remove self


def register_remover_hook(module: nn.Module, *handles: List[torch.utils.hooks.RemovableHandle]):
    """Convenience function that registers a hook that removes the given hooks and itself when called the first time."""
    own_hook_id = uuid.uuid4()
    remover_hook = partial(hook_remover_hook, handles=handles, own_hook_id=own_hook_id)
    remover_hook_handle = module.register_forward_hook(remover_hook)
    REMOVER_HOOK_HANDLES[own_hook_id] = remover_hook_handle
    return remover_hook_handle


def weight_norm(
    module: Union[nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear],
    init_mean: float = 0.0,
    init_scale: float = 1.0,
    name: str = "weight",
    data_dependent_init: bool = True,
    dim: Optional[int] = None,
    initializer: Callable = nn.init.kaiming_normal_,
    **initializer_kwargs: Dict[str, Any],
):
    """Wrap a module with weight normalization [1] including data dependent initialization.

    Args:
        module (Union[nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear]): Module to wrap
        init_mean (float, optional): Mean of the activations after weight normalization. Defaults to 0.0.
        init_scale (float, optional): Scale of the activations after weight normalization. Defaults to 1.0.
        name (str, optional): Name of the weight parameter to reparameterize. Defaults to "weight".
        data_dependent_init (bool, optional): Whether to use data dependent initialization. Defaults to True.
        dim (Optional[int], optional): Dimension of the weight parameter to norm along. Defaults to None.
        initializer (Callable, optional): Callable to initialize primary weight parameter of the module.
                                          Defaults to nn.init.kaiming_normal_.
        **initializer_kwargs (Dict[str, Any], optional): Keyword arguments to pass to initializer. Defaults to {}.

    Returns:
        nn.Module: Weight normalized module.

    [1] Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks \
        https://arxiv.org/abs/1602.07868
    """
    if dim is None:
        dim = 1 if getattr(module, "transposed", False) else 0

    module = nn.utils.weight_norm(module, dim=dim, name=name)

    if data_dependent_init:
        pre_hook = partial(wn_init_forward_pre_hook, name=name, dim=dim, initializer=initializer, **initializer_kwargs)
        post_hook = partial(wn_init_forward_hook, name=name, init_mean=init_mean, init_scale=init_scale)

        pre_hook_handle = module.register_forward_pre_hook(pre_hook)
        post_hook_handle = module.register_forward_hook(post_hook)

        register_remover_hook(module, pre_hook_handle, post_hook_handle)

    return module


def wn_init_forward_pre_hook(
    module: Union[nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear],
    input: torch.Tensor,
    name: str = "weight",
    dim: int = -1,
    initializer: Callable = nn.init.kaiming_normal_,
    **initializer_kwargs: Dict[str, Any],
):
    """Weight normalization [1] data-dependent initialization pre-hook that initializes the primary weight and bias.

    We update the normed weight parameter manually here (before the call to the forward hook). We do this because 
    `nn.utils.weight_norm` updates the normed weight at the call to forward method which actually occurs before 
    the pre-forward hook.

    The source of `_weight_norm` is found at:
    https://github.com/pytorch/pytorch/blob/c76c6e9bd345ba697ea2a6fb8c2e24f051bb2e00/torch/onnx/symbolic_opset9.py

    Args:
        module (Union[nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear]): Module to wrap
        input (Tuple[torch.Tensor]): Tuple of inputs to module.forward.
        initializer (Callable, optional): A callable to initialize primary weight parameter of the module. 
                                          Defaults to nn.init.kaiming_normal_.

    Returns:
        None: Does not alter the module input.

    [1] Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks \
        https://arxiv.org/abs/1602.07868
    """
    weight_v = getattr(module, name + "_v")
    weight_g = getattr(module, name + "_g")

    # initialize the weight
    initializer(weight_v, **initializer_kwargs)

    # pre-initialize the scale and bias
    nn.init.constant_(weight_g, 1.0)
    nn.init.constant_(module.bias, 0.0)

    # update normalized weight to reflect initialization
    weight = torch._weight_norm(weight_v, weight_g, dim=dim)
    setattr(module, name, weight)

    return input.detach()


def wn_init_forward_hook(
    module: Union[nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear],
    input: torch.Tensor,
    output: torch.Tensor,
    init_mean: float = 0.0,
    init_scale: float = 1.0,
    name: str = "weight",
):
    """Weight normalization [1] data-dependent initialization forward hook that initializes scale and bias.

    Args:
        module (Union[nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear]): Module to wrap
        input (Tuple[torch.Tensor]): Tuple of inputs to module.forward.
        output (torch.Tensor): Tensor of outputs from module.forward.
        init_mean (float, optional): Mean of the initial activations. Defaults to 0.0.
        init_scale (float, optional): Scale (standard deviation) of the initial activations. Defaults to 1.0.

    Raises:
        RuntimeError: If the input is not a batch, i.e. fewer than 1 dimension or only 1 batch example.

    Returns:
        torch.Tensor: Output as computed after data-dependent initialization. Has no gradients.

    [1] Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks \
        https://arxiv.org/abs/1602.07868
    """
    if input[0].ndim == 1 or not input[0].shape[0] > 1:
        raise RuntimeError(f"Cannnot do data-based WeightNorm initialization without a batch {input[0].shape=}")

    if module.bias is None:
        raise RuntimeError("Cannot do data-based WeightNorm initialization on a module without a bias parameter")

    if isinstance(module, nn.Linear):
        dim = list(range(0, input[0].ndim - 1))  # all but last dimension
    else:
        ndim = module.weight.ndim - 2  # 1D, 2D or 3D convolution
        dim = (0, *list(range(ndim + 1, input[0].ndim)))  # assumes channels first

    weight_g = getattr(module, name + "_g")

    with torch.no_grad():
        # compute statistics of output
        m_init, s_init = output.mean(dim, keepdim=True), output.std(dim, keepdim=True)

        scale_init = init_scale * (1 / s_init)
        scale_init_weight_shaped = scale_init.view(weight_g.size())
        bias_init = (-m_init * scale_init + init_mean).view(module.bias.size())

        weight_g.copy_(scale_init_weight_shaped)
        module.bias.copy_(bias_init)

        corrected_output = scale_init * (output - m_init) + init_mean

        return corrected_output
