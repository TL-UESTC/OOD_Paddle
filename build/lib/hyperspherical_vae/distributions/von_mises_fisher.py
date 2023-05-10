import math

import paddle


def pm(l, a, b):
    lt = [i for i in range(l)]
    num = lt[a]
    lt[a] = lt[b]
    lt[b] = num
    return lt

class VonMisesFisher(paddle.distribution.Distribution):
    def __init__(self, loc, scale, validata_args=None):
        self.dtype = loc.dtype
        self.loc = loc
        self.scale = scale
        self.__m = loc.shape[-1]
        self.__e1 = paddle.to_tensor([1.] + [0.] * (loc.shape[-1]-1))
        super(VonMisesFisher, self).__init__(self.loc.shape)
    
    def sample(self, shape=[]):
        with paddle.no_grad():
            return self.rsample(shape)
    
    def rsample(self, shape=[]):
        shape = shape if isinstance(shape, list) else [shape]
        w = self.__sample_w3(shape=shape) if self.__m == 3 else self.__sample_w_rej(shape=shape)
        
        temp1 = paddle.distribution.Normal(loc=0, scale=1).sample(shape=shape+self.loc.shape)
        perm = pm(len(temp1.shape), 0, -1)
        temp2 = temp1.transpose(perm=perm)[1:]
        perm = pm(len(temp2.shape), 0, -1)
        v = temp2.transpose(perm=perm)
        v = v / v.norm(axis=-1, keepdim=True)

        w_ = paddle.sqrt(paddle.clip(1 - (w ** 2), 1e-10))
        x = paddle.concat((w, w_ * v), axis=-1)
        z = self.__householder_rotation(x)

        return z.astype(self.dtype)

    def __sample_w3(self, shape):
        shape = shape + self.scale.shape
        u = paddle.distribution.Uniform(low=0, high=1).sample(shape=shape)
        self.__w = 1 + paddle.stack([paddle.log(u), paddle.log(1 - u) - 2 * self.scale], axis=0).logsumexp(axis=0) / self.scale
        return self.__w

    def __sample_w_rej(self, shape):
        c = paddle.sqrt((4 * (self.scale ** 2)) + (self.__m - 1) ** 2)
        b_true = (-2 * self.scale + c) / (self.__m - 1)
        b_app = (self.__m - 1) / (4 * self.scale)

        s = paddle.minimum(paddle.maximum(paddle.to_tensor([0.]), self.scale - 10), paddle.to_tensor([1.]))
        b = b_app * s + b_true * (1 - s)
        a = (self.__m - 1 + 2 * self.scale + c) / 4
        d = (4 * a * b) / (1 + b) - (self.__m - 1) * math.log(self.__m - 1)
        self.__b, (self.__e, self.__w) = b, self.__while_loop(b, a, d, shape)
        
        return self.__w
    
    def __while_loop(self, b, a, d, shape):
        b, a, d = [e.tile(repeat_times=shape + ([1]*len(self.scale.shape))) for e in (b, a, d)]
        w, e, bool_mask = paddle.zeros_like(b), paddle.zeros_like(b), (paddle.ones_like(b) == 1)
        w.stop_gradient = True
        e.stop_gradient = True
        shape = shape + self.scale.shape
        while bool_mask.sum() != 0:
            e_ = paddle.distribution.Beta(alpha=(self.__m - 1) / 2, beta=(self.__m - 1) / 2).sample(shape=shape[:-1]).reshape(shape=shape)
            u = paddle.distribution.Uniform(low=0, high=1).sample(shape=shape)

            w_ = (1 - (1 + b) * e_) / (1 - (1 - b) * e_)
            t = (2 * a * b) / (1 - (1 - b) * e_)

            accept = ((self.__m - 1) * t.log() - t + d) > paddle.log(u)
            reject = ~ accept

            w[bool_mask * accept] = w_[bool_mask * accept]
            e[bool_mask * accept] = e_[bool_mask * accept]

            temp_bool_mask = bool_mask.astype('int32')
            temp_reject = reject.astype('int32')
            temp_bool_mask[bool_mask * accept] = temp_reject[bool_mask * accept]
            bool_mask = temp_bool_mask.astype('bool')
        
        return e, w
    
    def __householder_rotation(self, x):
        u = self.__e1 - self.loc
        u = u / (u.norm(axis=-1, keepdim=True) + 1e-5)
        z = x - 2 * (x * u).sum(axis=-1, keepdim=True) * u 
        return z
