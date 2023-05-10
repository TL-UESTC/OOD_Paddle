import math
from math import ceil
from turtle import forward
import paddle
from paddle.fluid import dygraph
import paddle_torch.nn.functional as F
from . import functional
from paddle_torch.tensor import Tensor, convertTensor


'''Network'''


def forward_post_hook(layer, input, output):
    if isinstance(output, tuple):
        return tuple([convertTensor(x) if isinstance(x, dygraph.core.VarBase) else x for x in output])
    else:
        if isinstance(output, dygraph.core.VarBase) and not isinstance(output, Tensor):
            return convertTensor(output)
        else:
            return output

def forward_pre_hook(layer, input):
    if isinstance(input, tuple):
        return tuple([convertTensor(x) if isinstance(x, dygraph.core.VarBase) else x for x in input])
    else:
        if isinstance(input, dygraph.core.VarBase) and not isinstance(input, Tensor):
            return convertTensor(input)
        else:
            return input
            

class Module(paddle.nn.Layer):
    def __init__(self):
        super(Module, self).__init__(name_scope=None, dtype='float32')
        self.register_forward_post_hook(forward_post_hook)
        self.register_forward_pre_hook(forward_pre_hook)

    def load_state_dict(self, state_dict, strict=True):
        self.set_state_dict(state_dict, use_structured_name=True)
    
    def zero_grad(self):
        self.clear_gradients()

    @property
    def cuda(self):
        return self


class Conv2d(paddle.nn.Conv2D, Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', device=None, dtype=None):
        super(Conv2d, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=kernel_size, stride=stride,
                                     padding=padding, dilation=dilation, groups=groups,
                                     padding_mode='zeros', weight_attr=None, bias_attr=bias,
                                     data_format='NCHW')


class LeakyReLU(paddle.nn.LeakyReLU, Module):
    def __init__(self, negative_slope=0.01, inplace=False):
        super(LeakyReLU, self).__init__(negative_slope=negative_slope)
    #     self.negative_slope=negative_slope
    #     self.inplace = inplace
    
    # def forward(self, x):
    #     return F.leaky_relu(x, negative_slope=self.negative_slope, inplace=self.inplace)


class Linear(paddle.nn.Linear, Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None):
        super(Linear, self).__init__(in_features=in_features, out_features=out_features,
                                     weight_attr=None, bias_attr=bias, name=None)
        k = 1 / in_features
        self.weight = paddle.create_parameter(shape=self.weight.shape,
                                             dtype=str(self.weight.numpy().dtype),
                                             default_initializer=paddle.nn.initializer.Uniform(-math.sqrt(k), math.sqrt(k)))
        if bias:
            self.bias = paddle.create_parameter(shape=self.bias.shape,
                                                dtype=str(self.bias.numpy().dtype),
                                                default_initializer=paddle.nn.initializer.Uniform(-math.sqrt(k), math.sqrt(k)))


class LogSoftmax(paddle.nn.LogSoftmax, Module):
    def __init__(self, dim=-1):
        super(LogSoftmax, self).__init__(axis=dim)


class MaxPool2d(paddle.nn.MaxPool2D, Module):
    def __init__(self, kernel_size, stride=None, padding=0,
                 dilation=1, return_indices=False, ceil_mode=False):
        super(MaxPool2d, self).__init__(kernel_size=kernel_size,
                                        stride=stride, padding=padding,
                                        ceil_mode=ceil_mode,
                                        return_mask=return_indices,
                                        data_format='NCHW', name=None)


class ReLU(paddle.nn.ReLU, Module):
    def __init__(self, inplace=False):
        super(ReLU, self).__init__()
    #     self.inplace = inplace
    
    # def forward(self, x):
    #     return F.relu(x, inplace=self.inplace)


class Sequential(paddle.nn.Sequential, Module):
    def __init__(self, *args):
        super(Sequential, self).__init__()
        if len(args) > 0 and isinstance(args[0], (list, tuple)):
            for name, layer in args:
                self.add_sublayer(name, layer)
        else:
            for idx, layer in enumerate(args):
                self.add_sublayer(str(idx), layer)


'''Loss Function'''


class CrossEntropyLoss(paddle.nn.CrossEntropyLoss, Module):
    def __init__(self, weight=None, size_average=None,
                 ignore_index=-100, reduce=None,
                 reduction='mean', label_smoothing=0.0):
        super(CrossEntropyLoss, self).__init__(weight=weight,
                                               ignore_index=ignore_index,
                                               reduction=reduction,
                                               soft_label=False, axis=-1,
                                               use_softmax=True, name=None)


class L1Loss(paddle.nn.L1Loss, Module):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(L1Loss, self).__init__(reduction=reduction)


class NLLLoss(paddle.nn.NLLLoss, Module):
    def __init__(self, weight=None, size_average=None,
                 ignore_index=-100, reduce=None,
                 reduction='mean'):
        super(NLLLoss, self).__init__(weight=weight, ignore_index=ignore_index,
                                      reduction=reduction)


'''Others'''

def Parameter(data=None, fill_value=0.0, requires_grad=True):
    if isinstance(data, paddle.Tensor):
        # x = Parameter(data.shape, 0.0)
        # paddle.assign(data.astype('float32'), x)
        x = data.astype('float32')
    else:
        if isinstance(data, int):
            data = [data]
        x = paddle.create_parameter(shape=data, dtype='float32',
                                     attr=paddle.ParamAttr(name=None, initializer=paddle.nn.initializer.Constant(value=fill_value)),
                                     is_bias=False)
    if requires_grad:
        x.stop_gradient = False
    return convertTensor(x)


# class BatchNorm1d(paddle.nn.BatchNorm1D,Module):
#     def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None):
#         super(BatchNorm1d, self).__init__(num_features,  momentum=momentum, epsilon=eps)
    