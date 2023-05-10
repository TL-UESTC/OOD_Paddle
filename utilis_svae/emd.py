import paddle
import paddle.nn as nn


class SinkhornDistance(nn.Layer):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """
    def __init__(self, eps, max_iter, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction
    
    def forward(self, x, y):
        C= self._cost_matrix(x, y)
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]
        mu = paddle.empty(shape=[batch_size, x_points], dtype='float32').fill_(1.0 / x_points).squeeze()
        nu = paddle.empty(shape=[batch_size, y_points], dtype='float32').fill_(1.0 / y_points).squeeze()
        u = paddle.zeros_like(x=mu)
        v = paddle.zeros_like(x=nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u # useful to check the update
            
            x = self.M(C, u, v)
            u = self.eps * (paddle.log(mu+1e-8) - paddle.logsumexp(x=x, axis=-1)) + u
            
            x = self.M(C, u, v)
            perm = [i for i in range(len(x.shape))]
            temp = perm[-1]
            perm[-1] = perm[-2]
            perm[-2] = temp
            v = self.eps * (paddle.log(nu+1e-8) - paddle.logsumexp(x=x.transpose(perm=perm), axis=-1)) + v
            
            err = (u - u1).abs().sum(axis=-1).mean()
            actual_nits += 1
            if err.item() < thresh:
                break
        
        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = paddle.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = paddle.sum(pi * C, axis=(-2, -1))
        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()
        
        return cost, pi, C

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        x = (-C + u.unsqueeze(axis=-1) + v.unsqueeze(axis=-2)) / self.eps
        return x
    
    @staticmethod
    def _cost_matrix(x, y, p=2):
        x_col = x.unsqueeze(axis=-2)
        y_lin = y.unsqueeze(axis=-3)
        C = paddle.sum(x=(paddle.abs(x_col - y_lin)) ** p, axis=-1)
        return C
    