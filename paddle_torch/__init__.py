# from typing import Iterable
import numpy as np
import paddle
from paddle import static
from . import autograd
from . import tensor
from . import cuda
from . import distributions
from . import nn
from . import optim
from . import utils
from paddle_torch.tensor import Tensor, convertTensor, varbase_to_tensor
from collections.abc import Iterable

def abs(input):
    return convertTensor(paddle.abs(input))

def argmax(input, dim=None, keepdim=False):
    return convertTensor(paddle.argmax(input, axis=dim, keepdim=keepdim))

def cat(tensors, dim=0, out=None):
    x = paddle.concat(tensors, axis=dim)
    if out is None:
        return convertTensor(x)
    else:
        paddle.assign(x, out)
        return out

def clamp(input, min=None, max=None, *, out=None):
    return convertTensor(paddle.clip(input, min, max))

def device(name):
    if isinstance(name, int):
        if name<0:
            return paddle.CPUPlace()
        else:
            return paddle.CUDAPlace(int(name))

    if name.startswith('cuda'):
        device_id = name.replace('cuda', '').replace(':', '')
        if len(device_id)==0:
            return paddle.CUDAPlace(0)
        else:
            return paddle.CUDAPlace(int(device_id))
    else:
        return paddle.CPUPlace()

def dot(x, y):
    return convertTensor(paddle.dot(x, y))

def empty(*size, out=None, dtype=None, device=None, requires_grad=False, pin_memory=False):
    x = convertTensor(paddle.empty(shape=size, dtype=dtype))
    if requires_grad:
        x.stop_gradient = False
    if out is not None:
        out = x
    return x

def exp(x):
    return convertTensor(paddle.exp(x))

def from_numpy(x):
    return convertTensor(x)

def lgamma(input, *, out=None):
    x = convertTensor(paddle.lgamma(tensor(input)))
    if out is not None:
        out = x
    return x

def load(f, map_location=None, **pickle_load_args):
    return paddle.load(f)

def log(input, *, out=None):
    return convertTensor(paddle.log(tensor(input)))

def logsumexp(input, dim=None, keepdim=False, *, out=None):
    return convertTensor(paddle.logsumexp(tensor(input), axis=dim, keepdim=keepdim))

def manual_seed(seed):
    static.Program.random_seed = seed
    np.random.seed(seed)
    paddle.seed(seed)

def max(input, dim=None, keepdim=False):
    return convertTensor(paddle.max(input, axis=dim, keepdim=keepdim))

def maximum(input, other, *, out=None):
    x = convertTensor(paddle.maximum(x=input, y=other))
    if out is not None:
        out = x
    return x

def mean(input, dim=None, keepdim=False, *, dtype=None, out=None):
    x = convertTensor(paddle.mean(tensor(input), axis=dim, keepdim=keepdim))
    if out is None:
        return x
    else:
        paddle.assign(x, out)
        return out

def min(input, dim=None, keepdim=False):
    return convertTensor(paddle.min(input, axis=dim, keepdim=keepdim))

def minimum(input, other, *, out=None):
    x = convertTensor(paddle.minimum(x=input, y=other))
    if out is not None:
        out = x
    return x

def norm(input, p='fro', dim=None, keepdim=False, out=None, dtype=None):
    return convertTensor(paddle.norm(input, p=p, axis=dim, keepdim=keepdim))

def no_grad(func=None):
    return paddle.no_grad()

def ones(*size, out=None, dtype='float32', device=None, requires_grad=False):
    if isinstance(size[0], Iterable):
        size = size[0]
        if isinstance(size[0], Iterable):
            size = size[0]
    x =convertTensor(paddle.ones(size, dtype))
    if requires_grad:
        x.stop_gradient = False
    return x

def ones_like(input, *, dtype=None, layout=None, device=None, requires_grad=False):
    return convertTensor(paddle.ones_like(input, dtype=dtype))

def repeat(x, *size):
    if isinstance(size[0], Iterable):
        size = size[0]
    x = paddle.tile(x, size)
    return convertTensor(x)

def reshape(input, shape):
    return convertTensor(paddle.reshape(x=input, shape=shape))

def save(obj, f, pickle_protocol=2):
    paddle.save(obj, f, pickle_protocol)

def Size(size=[]):
    return size

def sqrt(x):
    return convertTensor(paddle.sqrt(tensor(x)))

def stack(tensors, dim=0, *, out=None):
    x = convertTensor(paddle.stack(x=tensor, axis=dim))
    if out is None:
        return x
    else:
        paddle.assign(x, out)
        return out

def sum(input, dim=None, keepdim=False, *, dtype=None):
    return convertTensor(paddle.sum(input, axis=dim, dtype=None, keepdim=keepdim))

def tensor(x, dtype=np.float32):
    if isinstance(x, list):
        x = paddle.to_tensor(x, dtype=dtype, stop_gradient=True)
    elif isinstance(x, int) or isinstance(x, np.int32):
        return convertTensor(convertTensor([x]).astype(dtype))
    return convertTensor(convertTensor(x).astype(dtype))

def Tensor(x, dtype=np.float32):
    return tensor(x, dtype)

def var(input, dim=None, unbiased=True, keepdim=False, *, out=None):
    x = convertTensor(paddle.var(tensor(input), axis=dim, unbiased=unbiased, keepdim=keepdim))
    if out is None:
        return x
    else:
        paddle.assign(x, out)
        return out

def zeros(*size, out=None, dtype='float32', device=None, requires_grad=True):
    if isinstance(size[0], Iterable):
        size = size[0]
        if isinstance(size[0], Iterable):
            size = size[0]
    x = convertTensor(paddle.zeros(size, dtype))
    if requires_grad:
        x.stop_gradient = False
    return x

def zeros_like(input, *, dtype=None, layout=None, device=None, requires_grad=False):
    x = convertTensor(paddle.zeros_like(input, dtype=dtype))
    if requires_grad:
        x.stop_gradient = False
    return x