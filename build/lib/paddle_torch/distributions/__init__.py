import paddle
from paddle_torch.tensor import convertTensor
from . import constraints
from . import kl
from . import normal


class Distribution(paddle.distribution.Distribution):
    '''
    The abstract base class for probability distributions. Functions are implemented in specific distributions.
    '''

    def __init__(self, batch_shape=(), event_shape=(), validate_args=None):
        super(Distribution, self).__init__(batch_shape=batch_shape, event_shape=event_shape)


class Beta(paddle.distribution.Beta):
    def __init__(self, concentration1, concentration0):
        super(Beta, self).__init__(concentration1, concentration0)
    
    def sample(self, sample_shape=()):
        return convertTensor(super(Beta, self).sample(sample_shape))

    def rsample(self, sample_shape=()):
        return convertTensor(self.sample(sample_shape))


class Normal(paddle.distribution.Normal):
    def __init__(self, loc, scale):
        super(Normal, self).__init__(loc, scale)

    def sample(self, sample_shape=[]):
        return convertTensor(super(Normal, self).sample(sample_shape))

    def rsample(self, sample_shape=[]):
        return convertTensor(self.sample(sample_shape))


class Uniform(paddle.distribution.Uniform):
    def __init__(self, low, high):
        super(Uniform, self).__init__(low, high)

    def sample(self, sample_shape=[]):
        return convertTensor(super(Uniform, self).sample(sample_shape))

    def rsample(self, sample_shape=[]):
        return convertTensor(self.sample(sample_shape))