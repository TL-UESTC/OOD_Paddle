import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class Attr_Encoder(nn.Layer):
    def __init__(self, input_size, mid_size, hidden_size):
        super(Attr_Encoder, self).__init__()
        self.fc1 = nn.Linear(input_size, mid_size, 
                             weight_attr=paddle.ParamAttr(initializer=nn.initializer.Uniform(low=-np.sqrt(1/input_size), high=np.sqrt(1/input_size))),
                             bias_attr=paddle.ParamAttr(initializer=nn.initializer.Uniform(low=-np.sqrt(1/input_size), high=np.sqrt(1/input_size))))
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(mid_size, hidden_size,
                             weight_attr=paddle.ParamAttr(initializer=nn.initializer.Uniform(low=-np.sqrt(1/mid_size), high=np.sqrt(1/mid_size))),
                             bias_attr=paddle.ParamAttr(initializer=nn.initializer.Uniform(low=-np.sqrt(1/mid_size), high=np.sqrt(1/mid_size))))
        self.fc3 = nn.Linear(mid_size, 1,
                             weight_attr=paddle.ParamAttr(initializer=nn.initializer.Uniform(low=-np.sqrt(1/mid_size), high=np.sqrt(1/mid_size))),
                             bias_attr=False)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x2 = self.fc2(x)
        x2 = x2 / x2.norm(axis=-1, keepdim=True)
        x3 = self.fc3(x)
        x3 = F.softplus(x3) + 100.0
        return x2, x3


class Attr_Decoder(nn.Layer):
    def __init__(self, hidden_size, mid_size, output_size):
        super(Attr_Decoder, self).__init__()
        self.fc1 = nn.Linear(hidden_size, mid_size,
                             weight_attr=paddle.ParamAttr(initializer=nn.initializer.Uniform(low=-np.sqrt(1/hidden_size), high=np.sqrt(1/hidden_size))),
                             bias_attr=paddle.ParamAttr(initializer=nn.initializer.Uniform(low=-np.sqrt(1/hidden_size), high=np.sqrt(1/hidden_size))))
        self.fc2 = nn.Linear(mid_size, output_size,
                             weight_attr=paddle.ParamAttr(initializer=nn.initializer.Uniform(low=-np.sqrt(1/mid_size), high=np.sqrt(1/mid_size))),
                             bias_attr=paddle.ParamAttr(initializer=nn.initializer.Uniform(low=-np.sqrt(1/mid_size), high=np.sqrt(1/mid_size))))
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class Encoder(nn.Layer):
    def __init__(self, input_size, mid_size, hidden_size):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_size, mid_size,
                             weight_attr=paddle.ParamAttr(initializer=nn.initializer.Uniform(low=-np.sqrt(1/input_size), high=np.sqrt(1/input_size))),
                             bias_attr=paddle.ParamAttr(initializer=nn.initializer.Uniform(low=-np.sqrt(1/input_size), high=np.sqrt(1/input_size))))
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(mid_size, hidden_size,
                             weight_attr=paddle.ParamAttr(initializer=nn.initializer.Uniform(low=-np.sqrt(1/mid_size), high=np.sqrt(1/mid_size))),
                             bias_attr=paddle.ParamAttr(initializer=nn.initializer.Uniform(low=-np.sqrt(1/mid_size), high=np.sqrt(1/mid_size))))
        self.fc3 = nn.Linear(mid_size, 1,
                             weight_attr=paddle.ParamAttr(initializer=nn.initializer.Uniform(low=-np.sqrt(1/mid_size), high=np.sqrt(1/mid_size))),
                             bias_attr=False)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu2(x)
        x2 = self.fc2(x)
        x2 = x2 / x2.norm(axis=-1, keepdim=True)
        x3 = self.fc3(x)
        x3 = F.softplus(x3) + 100.0
        return x2, x3


class Decoder(nn.Layer):
    def __init__(self, hidden_size, mid_size, output_size):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(hidden_size, mid_size,
                             weight_attr=paddle.ParamAttr(initializer=nn.initializer.Uniform(low=-np.sqrt(1/hidden_size), high=np.sqrt(1/hidden_size))),
                             bias_attr=paddle.ParamAttr(initializer=nn.initializer.Uniform(low=-np.sqrt(1/hidden_size), high=np.sqrt(1/hidden_size))))
        self.fc2 = nn.Linear(mid_size, output_size,
                             weight_attr=paddle.ParamAttr(initializer=nn.initializer.Uniform(low=-np.sqrt(1/mid_size), high=np.sqrt(1/mid_size))),
                             bias_attr=paddle.ParamAttr(initializer=nn.initializer.Uniform(low=-np.sqrt(1/mid_size), high=np.sqrt(1/mid_size))))    
        self.relu1 = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x


class Decoder_Imagenet(nn.Layer):
    def __init__(self, hidden_size, mid_size, output_size):
        super(Decoder_Imagenet, self).__init__()
        self.fc1 = nn.Linear(hidden_size, mid_size,
                             weight_attr=paddle.ParamAttr(initializer=nn.initializer.Uniform(low=-np.sqrt(1/hidden_size), high=np.sqrt(1/hidden_size))),
                             bias_attr=paddle.ParamAttr(initializer=nn.initializer.Uniform(low=-np.sqrt(1/hidden_size), high=np.sqrt(1/hidden_size))))
        self.fc2 = nn.Linear(mid_size, output_size,
                             weight_attr=paddle.ParamAttr(initializer=nn.initializer.Uniform(low=-np.sqrt(1/mid_size), high=np.sqrt(1/mid_size))),
                             bias_attr=paddle.ParamAttr(initializer=nn.initializer.Uniform(low=-np.sqrt(1/mid_size), high=np.sqrt(1/mid_size))))
        self.relu1 = nn.LeakyReLU(0.2)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x


class LINEAR_LOGSOFTMAX(nn.Layer):
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX, self).__init__()
        self.fc = nn.Linear(input_dim, nclass,
                            weight_attr=paddle.ParamAttr(initializer=nn.initializer.Uniform(low=-np.sqrt(1/input_dim), high=np.sqrt(1/input_dim))),
                            bias_attr=paddle.ParamAttr(initializer=nn.initializer.Uniform(low=-np.sqrt(1/input_dim), high=np.sqrt(1/input_dim))))
        self.logic = nn.LogSoftmax(axis=1)

    def forward(self, x):
        o = self.logic(self.fc(x))
        return o
