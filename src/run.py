from torch import (
    Tensor,
    min as tmin,
    max as tmax
)
from torch.nn import (
    Module,
    MSELoss,
    KLDivLoss,
    L1Loss
)
from torch.optim import SGD
from pandas import DataFrame, concat
from matplotlib import pyplot
from math import (
    sin,
    cos,
    pi
)
from sklearn.preprocessing import RobustScaler
from numpy import array

from data import (
    perfect_circle_data,
    fuzzy_circle_data
)
from models import (
    CircleNetwork,
    ModelTrainer
)

radius = 10
step = 0.01
val_data = perfect_circle_data(radius,step)
train_data = fuzzy_circle_data(radius,step)
num_epochs = 10000

x_train = train_data[["x","y"]].to_numpy()
y_train = array(train_data["theta"].to_numpy()).reshape(-1,1)
x_val = val_data[["x","y"]].to_numpy()
y_val = array(val_data["theta"].to_numpy()).reshape(-1,1)

x_scale = RobustScaler()
y_scale = RobustScaler()

s_x_train = Tensor(x_scale.fit_transform(x_train))
s_y_train = Tensor(y_scale.fit_transform(y_train))
s_x_val = Tensor(x_scale.transform(x_val))
s_y_val = Tensor(y_scale.transform(y_val))

circle_net = CircleNetwork(2,1,200,0.5)
opt = SGD(circle_net.parameters(), lr=1e-5)
loss = MSELoss(reduction="sum")
trainer = ModelTrainer()

train_losses, val_losses = trainer.train(circle_net,
    opt,
    loss,
    num_epochs,
    s_x_train,
    s_y_train,
    s_x_val,
    s_y_val
)

eval_data = fuzzy_circle_data(radius,step)
x_eval = eval_data[["x","y"]].to_numpy()
y_eval = array(eval_data["theta"].to_numpy()).reshape(-1,1)
s_x_eval = Tensor(x_scale.fit_transform(x_eval))
y_scale.fit(y_eval)
s_y_pred = circle_net(s_x_eval)
y_pred = y_scale.inverse_transform(s_y_pred.detach().numpy())

pred_thetas_list = y_pred.flatten().tolist()
pred_x = []
pred_y = []
for theta in pred_thetas_list:
    y = radius * sin(theta * (pi / 180))
    x = radius * cos(theta * (pi / 180))
    pred_x.append(x)
    pred_y.append(y)

pyplot.subplot(3,2,1)
pyplot.scatter(train_data["x"], train_data["y"])
pyplot.xlabel("x train")
pyplot.ylabel("y train")
pyplot.title("training data")
pyplot.subplot(3,2,2)
pyplot.scatter(val_data["x"], val_data["y"])
pyplot.xlabel("x val")
pyplot.ylabel("y val")
pyplot.title("validation data")
pyplot.subplot(3,2,3)
pyplot.scatter(eval_data["x"], eval_data["y"])
pyplot.xlabel("x eval")
pyplot.ylabel("y eval")
pyplot.title("evaluation data")
pyplot.subplot(3,2,4)
pyplot.scatter(pred_x, pred_y)
pyplot.xlabel("x predicted")
pyplot.ylabel("y predicted")
pyplot.title("predictions")
pyplot.subplot(3,2,5)
pyplot.scatter(val_data["theta"], y_pred)
pyplot.xlabel("true angle")
pyplot.ylabel("predicted angle")
pyplot.title("true vs predicted")
pyplot.subplot(3,2,6)
pyplot.scatter(range(1,len(train_losses)+1), val_losses)
pyplot.xlabel("epoch")
pyplot.ylabel("error")
pyplot.title("error")
pyplot.show()