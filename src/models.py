from torch import (
    Tensor,
    no_grad
)
from torch.nn import (
    Module,
    Linear,
    Dropout,
    Hardtanh,
    LeakyReLU,
    Softsign
)
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from pandas import DataFrame

class CircleNetwork(Module):
    def __init__(self,
        num_in_cols: int,
        num_out_cols: int,
        hide_nodes: int,
        dropout_prob: float
    ):
        super(CircleNetwork, self).__init__()
        self.lin1 = Linear(num_in_cols, hide_nodes)
        self.lin2 = Linear(hide_nodes, hide_nodes)
        self.lin3 = Linear(hide_nodes, hide_nodes)
        self.lin4 = Linear(hide_nodes, hide_nodes)
        self.lin5 = Linear(hide_nodes, 1)
        self.leak_act = LeakyReLU()
        self.tanh_act = Hardtanh()
        self.soft_act = Softsign()
        self.drop = Dropout(dropout_prob)
    
    def forward(self, x):
        #linear predictions 1
        lin_out = self.lin1(x)
        act_out = self.leak_act(lin_out)
        drop_out = self.drop(act_out)
        #linear predictions 2
        lin_out = self.lin2(drop_out)
        act_out = self.soft_act(lin_out)
        drop_out = self.drop(act_out)
        #linear predictions 3
        lin_out = self.lin3(drop_out)
        act_out = self.tanh_act(act_out)
        drop_out = self.drop(act_out)
        #linear predictions 4
        lin_out = self.lin4(drop_out)
        act_out = self.leak_act(lin_out)
        drop_out = self.drop(act_out)
        #final linear predictions
        return self.lin5(drop_out)

class ModelTrainer(Module):
    def __init__(self):
        super(ModelTrainer, self).__init__()
    
    def train(self,
        model: Module,
        opt: Optimizer,
        loss_func: _Loss,
        num_epochs: int,
        x_train: Tensor,
        y_train: Tensor,
        x_val: Tensor,
        y_val: Tensor
    ):
        train_losses = []
        val_losses = []
        for epoch in range(0, num_epochs+1):
            #training data
            model.train()
            y_pred = model(x_train)
            train_loss = loss_func(y_pred, y_train)
            train_losses.append(train_loss.item())
            opt.zero_grad()
            train_loss.backward()
            opt.step()
            #validation data
            model.eval()
            with no_grad():
                y_pred = model(x_val)
                val_loss = loss_func(y_pred, y_val)
                val_losses.append(val_loss.item())
            print(f"epoch {epoch}: train loss {train_loss.item()} ;;; val loss {val_loss.item()}")
        return train_losses, val_losses
        