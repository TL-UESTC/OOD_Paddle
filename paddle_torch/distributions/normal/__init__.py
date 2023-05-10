import paddle
from paddle_torch.tensor import convertTensor

# def Normal(loc, scale):
#     return paddle.distribution.Normal(loc, scale)

class Normal(paddle.distribution.Normal):
    def __init__(self, loc, scale):
        super(Normal, self).__init__(loc, scale)
    
    def sample(self, sample_shape=[]):
        return convertTensor(super(Normal, self).sample(sample_shape))

    def rsample(self, sample_shape=[]):
        return convertTensor(self.sample(sample_shape))