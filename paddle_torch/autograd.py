import paddle
from paddle.fluid import core
from collections import defaultdict


class Function(paddle.autograd.PyLayer):
    def __init__(self):
        super(Function, self).__init__()