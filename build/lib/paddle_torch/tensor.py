import paddle
import paddle.fluid as fluid
from paddle.fluid import dygraph
import numpy as np
from typing import Iterable
import paddle_torch as ptorch

enable_monkeypatch = False

def convertTensor(x):
    if enable_monkeypatch:
        if isinstance(x, paddle.Tensor):
            return x
    if isinstance(x, Tensor):
        return x
    # ret = Tensor(paddle.to_tensor(x))
    ret = Tensor(x)
    return ret

def new_full(size, fill_value, dtype=None, requires_grad=False):
    if dtype is None:
        dtype = 'float32'
    x = convertTensor(paddle.full(size, fill_value, dtype=dtype))
    x.stop_gradient = not requires_grad
    return x

def varbase_to_tensor(x):
    return convertTensor(x)


class Tensor(paddle.Tensor):
    def __init__(self, *args, **kwargs):
        if isinstance(args[0], dygraph.core.VarBase) or isinstance(args[0], dygraph.core.LoDTensor):
            dtype = args[0].dtype
            super(Tensor, self).__init__(dtype, args[0].shape, args[0].name, dygraph.core.VarDesc.VarType.LOD_TENSOR, True)
            fluid.layers.assign(args[0], self)
        elif isinstance(args[0], Iterable):
            args = list(args)
            if isinstance(args[0][0], int):
                args[0] = np.array(args[0]).astype('int32')
            else:
                args[0] = np.array(args[0]).astype('float32')
            super(Tensor, self).__init__(*args, **kwargs)
        elif isinstance(args[0], int):
            super(Tensor, self).__init__(np.zeros(args).astype('float32'))
        else:
            super(Tensor, self).__init__(*args, **kwargs)

    @property
    def data(self):
        return convertTensor(self)

    @property
    def device(self):
        return str(self.place)

    @property
    def grad(self):
        if getattr(self, 'grad_orig', None) is None:
            if super(Tensor, self).grad is None:
                return None
            return convertTensor(super(Tensor, self).grad)
        else:
            return self.grad_orig

    @property
    def is_cuda(self):
        if 'cuda' in str(self.place):
            return True
        else:
            return False
    
    @property
    def T(self):
        return self.t()

    def argmax(self, dim=None, keepdim=False):
        return ptorch.argmax(self, dim=dim, keepdim=keepdim)

    def astype(self, dtype=None):
        return convertTensor(super(Tensor, self).astype(dtype))

    def cuda(self, *args, **kwargs):
        return self

    def detach(self):
        return convertTensor(super(Tensor, self).detach())

    def dim(self):
        return len(self.shape)

    def fill_(self, value):
        x = convertTensor(paddle.full(shape=self.shape, fill_value=value, dtype=self.dtype))
        paddle.assign(x, self)
        return self

    def float(self):
        return convertTensor(self.astype('float32'))

    def long(self):
        return convertTensor(self.astype('int64'))

    def logsumexp(self, dim=None, keepdim=False):
        return ptorch.logsumexp(self, dim=dim, keepdim=keepdim)

    def max(self, dim=None, keepdim=False):
        return ptorch.max(self, dim=dim, keepdim=keepdim)

    def mean(self, dim=None, keepdim=False, *, dtype=None):
        return ptorch.mean(self, dim=dim, keepdim=keepdim)

    def norm(self, p='fro', dim=None, keepdim=False, dtype=None):
        return ptorch.norm(self, p=p, dim=dim, keepdim=keepdim, dtype=dtype)

    def permute(self, *perm):
        perm = [len(perm)+x if x<0 else x for x in perm]
        x = paddle.transpose(self, perm)
        return convertTensor(x)

    def repeat(self, *size):
        return convertTensor(ptorch.repeat(self, size))

    def reshape(self, *shape):
        if len(shape) == 1:
            shape = shape[0]
        return self.view(*shape)

    def size(self, dim=None):
        if dim is None:
            return self.shape
        return self.shape[dim]
    
    def sum(self, dim=None, keepdim=False, dtype=None):
        return convertTensor(ptorch.sum(self, dim=dim, keepdim=keepdim, dtype=dtype))

    def squeeze(self, dim=[]):
        if isinstance(dim, int):
            dim = [dim]
        x = paddle.squeeze(self, dim)
        return convertTensor(x)

    def t(self):
        return convertTensor(paddle.t(self))

    def tile(self, repeat_times):
        return convertTensor(super(Tensor, self).tile(repeat_times))

    def to(self, *args, **kwargs):
        if 'dtype' in kwargs:
            dtype = str(kwargs['dtype'])
        elif isinstance(args[0], paddle.Tensor):
            dtype = str(args[0].dtype)
            if '64' in dtype:
                dtype = 'int32'
            elif '32' in dtype:
                dtype = 'float32'
            else:
                # return self
                return convertTensor(self)
        else:
            dtype=str(args[0])
        if dtype == 'int32':
            # return self.long()
            return convertTensor(self.long())
        elif dtype == 'float32':
            # return self.float()
            return convertTensor(self.float())
        # return self
        return convertTensor(self)

    def transpose(self, *perm):
        if isinstance(perm[0], Iterable):
            return convertTensor(paddle.transpose(self, perm[0]))
        perm2 = list(range(len(self.shape)))
        a = perm2[perm[0]]
        perm2[perm[0]] = perm[1]
        perm2[perm[1]] = a
        perm = perm2
        return self.permute(*perm)

    def type(self, dtype):
        return self.astype(dtype)

    def uniform_(self, low=0, high=1):
        paddle.assign(convertTensor(paddle.uniform(self.shape, dtype='float32', min=low, max=high, seed=0)), self)
        return self

    def unsqueeze(self, dim):
        x = paddle.unsqueeze(self, dim)
        return convertTensor(x)

    def var(self, dim=None, unbiased=True, keepdim=False):
        return ptorch.var(self, dim=dim, unbiased=unbiased, keepdim=keepdim)

    def view(self, *size):
        if len(size)==1:
            if isinstance(size[0], Iterable):
                size = size[0]
        x = paddle.reshape(self, size)
        return convertTensor(x)

    def __add__(self, other):
        return convertTensor(super(Tensor, self).__add__(other))
    
    def __eq__(self, other):
        return convertTensor(super(Tensor, self).__eq__(other))

    def __div__(self, other_var):
        return convertTensor(super(Tensor, self).__div__(other_var))

    def __invert__(self):
        return convertTensor(paddle.logical_not(self))

    #两个输入矩阵的相乘
    def __matmul__(self, other):
        return convertTensor(paddle.mm(self.float(), other.float()))

    def __mod__(self, other):
        return convertTensor(super(Tensor, self).__mod__(other))

    def __mul__(self, other_var):
        return convertTensor(super(Tensor, self).__mul__(other_var))
    
    def __or__(self, other):
        return convertTensor(paddle.logical_or(self, other))

    def __pow__(self, other):
        return convertTensor(super(Tensor, self).__pow__(other))

    def __sub__(self, other):
        return convertTensor(super(Tensor, self).__sub__(other))

    def __truediv__(self, other_var):
        return self.__div__(other_var)

    def __getitem__(self, args):
        if isinstance(args, tuple):
            if isinstance(args[0], Iterable) and args[1]==slice(None, None, None):
                if len(args[0])==0:
                    return convertTensor(paddle.to_tensor([]))
                if isinstance(args[0][0], bool) or isinstance(args[0][0], np.bool_):
                    if len(self.shape) > 1 and self.shape[0] == len(args[0]):
                        result = []
                        temp = self.numpy().tolist()
                        for i, j in enumerate(args[0]):
                            if j:
                                result.append(temp[i])
                        return convertTensor(paddle.to_tensor(result))
                    else:
                        raise IndexError
                else:
                    x=convertTensor(args[0]).astype('int32')
                    return convertTensor(paddle.index_select(self,x,axis=0)).astype(self.dtype)
            if args[0]==slice(None, None, None) and isinstance(args[1], Iterable):
                if len(args[1])==0:
                    return convertTensor(paddle.to_tensor([]))
                if isinstance(args[1][0], bool) or isinstance(args[1][0], np.bool_):
                    if self.shape[1] == len(args[1]):
                        result = np.array([[] for i in range(self.shape[0])])
                        temp = self.numpy()
                        for i, j in enumerate(args[1]):
                            if j:
                                result = np.column_stack((result, temp[:, i]))
                        return convertTensor(paddle.to_tensor(result))
                    else:
                        raise IndexError
                else:
                    x=convertTensor(args[1]).astype('int32')
                    return convertTensor(paddle.index_select(self,x,axis=1)).astype(self.dtype)
        if isinstance(args, np.ndarray):
            if len(args)==0:
                return convertTensor(paddle.to_tensor([]))
            if isinstance(args[0], np.bool_):
                if len(self.shape) > 1 and self.shape[0] == len(args):
                        result = []
                        temp = self.numpy().tolist()
                        for i, j in enumerate(args):
                            if j:
                                result.append(temp[i])
                        return convertTensor(paddle.to_tensor(result))
                else:
                    raise IndexError
            else:
                args=convertTensor(args).astype('int32')
                return convertTensor(paddle.index_select(self,args,axis=0))
        if isinstance(args, list):
            if len(args)==0:
                return convertTensor(paddle.to_tensor([]))
            if isinstance(args[0], bool):
                if len(self.shape) > 1 and self.shape[0] == len(args):
                        result = []
                        temp = self.numpy().tolist()
                        for i, j in enumerate(args):
                            if j:
                                result.append(temp[i])
                        return convertTensor(paddle.to_tensor(result))
                else:
                    raise IndexError
            else:
                args=convertTensor(args).astype('int32')
                return convertTensor(paddle.index_select(self,args,axis=0))
        if isinstance(args, Iterable):
            args2 = list(args)
            for j in range(len(args)):
                k = len(args) - 1
                if args[k] is None:
                    self.unsqueeze_(axis=k)
                    args2[k] = slice(None, None, None)
            args = tuple(args2)
        if getattr(self, '__getitem__origin', None) is None:
            return convertTensor(super(Tensor, self).__getitem__(args))
        else:
            return self.__getitem__origin(args)