from re import X
import paddle.nn.functional as F
import paddle_torch as ptorch
from paddle_torch.tensor import convertTensor

def cross_entropy(input, target, weight=None,
                  size_average=None, ignore_index=- 100,
                  reduce=None, reduction='mean',
                  label_smoothing=0.0):
    return F.cross_entropy(input=ptorch.tensor(input), label=target, weight=weight,
                           ignore_index=ignore_index, reduction=reduction,
                           soft_label=False, axis=-1)

def cosine_similarity(x1, x2, dim=1, eps=1e-08):
    return convertTensor(F.cosine_similarity(x1=ptorch.tensor(x1), x2=ptorch.tensor(x2), axis=dim, eps=eps))

def leaky_relu(input, negative_slope=0.01, inplace=False):
    x = convertTensor(F.leaky_relu(ptorch.tensor(input), negative_slope=negative_slope))
    if inplace==False:
        return x
    else:
        input = x
        return input

def log_softmax(input, dim=-1, _stacklevel=3, dtype=None):
    return convertTensor(F.log_softmax(ptorch.tensor(input), axis=dim, dtype=dtype))

def relu(input, inplace=False):
    if inplace==False:
        return convertTensor(F.relu(ptorch.tensor(input), name=None))
    else:
        return convertTensor(F.relu_(ptorch.tensor(input), name=None))

def softmax(input, dim=0, _stacklevel=3, dtype=None):
    return convertTensor(F.softmax(ptorch.tensor(input), axis=dim, dtype=dtype))

def softplus(input, beta=1, threshold=20):
    return convertTensor(F.softplus(ptorch.tensor(input), beta=beta, threshold=threshold))