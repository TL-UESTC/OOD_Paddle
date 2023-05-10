import paddle

def is_available():
    try:
        paddle.CUDAPlace(0)
        return True
    except:
        return False