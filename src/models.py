from torch.nn import (
    Module,
    Linear,
    Tanh,
    Dropout
)
from torch.nn.modules.loss import _Loss
from torch.optim import (
    Optimizer
)
from pandas import DataFrame

class CircleNetwork(Module):
    def __init__(self, num_in_cols: int, num_out_cols: int, hide_nodes: int, dropout_prob: float):
        super(CircleNetwork, self).__init__()
        self.lin1 = Linear(num_in_cols, hide_nodes)
        self.lin2 = Linear(hide_nodes, 1)
        self.act_tanh = Tanh()
        self.drop = Dropout(dropout_prob)
    
    def forward(self, x):
        lin1_out = self.lin1(x)
        tanh_out = self.act_tanh(lin1_out)
        drop_out = self.drop(tanh_out)
        return self.lin2(drop_out)

class ModelTrainer(Module):
    def __init__(self, opt: Optimizer, loss: _Loss):
        super(ModelTrainer, self).__init__()
        self.opt = opt
        self.loss = loss
    
    def train(self, model: Module, num_epochs: int, x_train, y_train, x_val, y_val):
        