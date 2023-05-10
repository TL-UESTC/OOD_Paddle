import paddle.distribution

def register_kl(p, q):
    return paddle.distribution.register_kl(cls_p=p, cls_q=q)