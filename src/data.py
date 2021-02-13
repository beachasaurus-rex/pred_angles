from math import (
    sqrt,
    pi,
    asin,
    copysign
)
from pandas import (
    concat,
    Series,
    DataFrame
)
from numpy import (
    linspace as nplinspace,
    array,
    concatenate
)
from numpy.random import normal
from math import ceil

def _circle(x: float, radius: int):
    return sqrt(radius**2 - x**2)

def _theta(x: float, y: float):
    r_mag = sqrt(x**2 + y**2)
    x_sign = None
    y_sign = None
    if x == 0:
        x_sign = 1
    else:
        x_sign = copysign(1,x)
    if y == 0:
        y_sign = 1
    else:
        y_sign = copysign(1,y)
    theta = asin(y / r_mag) * (180 / pi)
    if x_sign == -1 and y_sign == 1:
        theta = 180 - theta
    elif x_sign == -1 and y_sign == -1:
        theta = 180 - theta
    elif x_sign == 1 and y_sign == -1:
        theta = 360 + theta
    return theta

def _num_eles(start: float, stop: float, step:float):
    return ceil(abs(stop-start)/step) + 1

def _add_noise(eles: list, radius: int):
    r = normal(0,0.1,len(eles))
    new_eles = [eles[i] * (1 + r[i]) for i in range(0, len(eles))]
    return new_eles

def make_validation_data(radius: float, step: float):
    #half circle 1
    num_eles = _num_eles(radius,-radius,step)
    x1 = nplinspace(radius, -radius, num_eles)
    y1 = array([_circle(x_iter, radius) for x_iter in x1])
    #half circle 2
    num_eles = _num_eles(-(radius-step),radius,step)
    x2 = nplinspace(-(radius-step), radius, num_eles)
    y2 = array([-_circle(x_iter, radius) for x_iter in x2])
    #concat everything
    x = concatenate([x1, x2], axis=0)
    y = concatenate([y1, y2], axis=0)
    #calc the angle for each coordinate
    theta = array([_theta(x[i],y[i]) for i in range(0,len(x))])
    df = concat([Series(x),Series(y),Series(theta)], axis=1)
    df.columns = ["x", "y", "theta"]
    return df

def make_training_data(radius: float, step: float):
    #half circle 1
    num_eles = _num_eles(radius,-radius,step)
    x1 = nplinspace(radius, -radius, num_eles)
    y1 = array([_circle(x_iter, radius) for x_iter in x1])
    #half circle 2
    num_eles = _num_eles(-(radius-step),radius,step)
    x2 = nplinspace(-(radius-step), radius, num_eles)
    y2 = array([-_circle(x_iter, radius) for x_iter in x2])
    #concat everything
    x = concatenate([x1, x2], axis=0)
    y = concatenate([y1, y2], axis=0)
    #calc the angle for each coordinate
    theta = array([_theta(x[i],y[i]) for i in range(0,len(x))])
    #add some noise
    x = _add_noise(x, radius)
    y = _add_noise(y, radius)
    theta = _add_noise(theta, radius)
    df = concat([Series(x),Series(y),Series(theta)], axis=1)
    df.columns = ["x", "y", "theta"]
    return df
