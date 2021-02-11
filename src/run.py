from tensorflow.keras import (
    layers,
    Sequential
)
from pandas import DataFrame, concat
from matplotlib import pyplot
from math import (
    sin,
    cos,
    pi
)

from data import (
    make_validation_data,
    make_training_data,
    scale,
    unscale
)

radius = 100
val_data = make_validation_data(radius,1)
train_data = make_training_data(radius,1)
num_epochs = 100

x_min = train_data["x"].min()
x_max = train_data["x"].max()
y_min = train_data["y"].min()
y_max = train_data["y"].max()
theta_min = train_data["theta"].min()
theta_max = train_data["theta"].max()
s_x = scale(train_data["x"], x_min, x_max)
s_y = scale(train_data["y"], y_min, y_max)
s_theta = scale(train_data["theta"], theta_min, theta_max)
s_input = concat([s_x,s_y], axis=1)

val_x_min = val_data["x"].min()
val_x_max = val_data["x"].max()
val_y_min = val_data["y"].min()
val_y_max = val_data["y"].max()
val_theta_min = val_data["theta"].min()
val_theta_max = val_data["theta"].max()
val_s_x = scale(val_data["x"], val_x_min, val_x_max)
val_s_y = scale(val_data["y"], val_y_min, val_y_max)
val_s_theta = scale(val_data["theta"], val_theta_min, val_theta_max)
val_s_input = concat([val_s_x, val_s_y], axis=1)

model = Sequential([
    layers.Dense(20, activation="relu", input_shape=[2]),
    layers.Dense(20, activation="relu"),
    layers.Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mae"
)

history = model.fit(
    s_input, s_theta,
    validation_data=(val_s_input, val_s_theta),
    epochs=num_epochs
)

test_data = make_training_data(radius,1)

test_x_min = test_data["x"].min()
test_x_max = test_data["x"].max()
test_y_min = test_data["y"].min()
test_y_max = test_data["y"].max()
test_theta_min = test_data["theta"].min()
test_theta_max = test_data["theta"].max()
test_s_x = scale(test_data["x"], test_x_min, test_x_max)
test_s_y = scale(test_data["y"], test_y_min, test_y_max)
test_s_input = concat([test_s_x, test_s_y], axis=1)

s_pred_thetas = model.predict(s_input)
pred_thetas = unscale(s_pred_thetas, test_theta_min, test_theta_max)

pred_thetas_list = pred_thetas.flatten().tolist()
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
pyplot.scatter(test_data["x"], test_data["y"])
pyplot.xlabel("x eval")
pyplot.ylabel("y eval")
pyplot.title("evaluation data")
pyplot.subplot(3,2,4)
pyplot.scatter(pred_x, pred_y)
pyplot.xlabel("x predicted")
pyplot.ylabel("y predicted")
pyplot.title("predictions")
pyplot.subplot(3,2,5)
pyplot.scatter(val_data["theta"], pred_thetas)
pyplot.xlabel("true angle")
pyplot.ylabel("predicted angle")
pyplot.title("true vs predicted")
pyplot.show()