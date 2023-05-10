import paddle
from . import lr_scheduler
from typing import Optional

class Adam(paddle.optimizer.Adam):
    def __init__(self, params=None, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=None, amsgrad=False, *, foreach: Optional[bool] = None,
                 maximize: bool = False, capturable: bool = False):
        super(Adam, self).__init__(learning_rate=lr, beta1=betas[0],
                                   beta2=betas[1], epsilon=eps, parameters=params,
                                   weight_decay=weight_decay, grad_clip=None,
                                   name=None, lazy_mode=False)

    def zero_grad(self):
        self.clear_gradients()


class SGD(paddle.optimizer.SGD):
    def __init__(self, params, lr=1e-3, momentum=0, dampening=0,
                 weight_decay=None, nesterov=False, *, maximize=False, 
                 foreach: Optional[bool] = None):
        super(SGD, self).__init__(learning_rate=lr, parameters=params,
                                  weight_decay=weight_decay, grad_clip=None,
                                  name=None)

    def zero_grad(self):
        self.clear_gradients()
