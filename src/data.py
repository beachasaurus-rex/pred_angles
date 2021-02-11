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
from numpy.random import normal

def _circle(x: float, radius: int):
    return sqrt(radius**2 - x**2)

def _mag(x: float, y: float):
    return sqrt(x**2 + y**2)

def _theta(x: float, y: float):
    r_mag = _mag(x,y)
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

def make_validation_data(radius: int, step: int):
    #quad 1
    q1x = [i for i in range(radius,-1,-step)]
    q1y = [_circle(x_iter, radius) for x_iter in q1x]
    #quad 2
    q2x = [i for i in range(-1,-(radius+1),-step)]
    q2y = [_circle(x_iter, radius) for x_iter in q2x]
    #quad 3
    q3x = [i for i in range(-(radius-1),1,step)]
    q3y = [-_circle(x_iter, radius) for x_iter in q3x]
    #quad 4
    q4x = [i for i in range(1,radius+1,step)]
    q4y = [-_circle(x_iter, radius) for x_iter in q4x]
    #concat everything
    x = [*q1x,*q2x,*q3x,*q4x]
    y = [*q1y,*q2y,*q3y,*q4y]
    #calc the angle for each coordinate
    theta = [_theta(x[i],y[i]) for i in range(0,len(x))]
    df = concat([Series(x),Series(y),Series(theta)], axis=1)
    df.columns = ["x", "y", "theta"]
    return df

def _add_noise(eles: list, radius: int):
    r = normal(0,0.1,len(eles))
    new_eles = [eles[i] * (1 + r[i]) for i in range(0, len(eles))]
    return new_eles

def make_training_data(radius: int, step: int):
    #quad 1
    q1x = [i for i in range(radius,-1,-step)]
    q1y = [_circle(x_iter, radius) for x_iter in q1x]
    #quad 2
    q2x = [i for i in range(-1,-(radius+1),-step)]
    q2y = [_circle(x_iter, radius) for x_iter in q2x]
    #quad 3
    q3x = [i for i in range(-(radius-1),1,step)]
    q3y = [-_circle(x_iter, radius) for x_iter in q3x]
    #quad 4
    q4x = [i for i in range(1,radius+1,step)]
    q4y = [-_circle(x_iter, radius) for x_iter in q4x]
    #concat everything
    x = [*q1x,*q2x,*q3x,*q4x]
    y = [*q1y,*q2y,*q3y,*q4y]
    #calc the angle for each coordinate
    theta = [_theta(x[i],y[i]) for i in range(0,len(x))]
    #add some noise
    x = _add_noise(x, radius)
    y = _add_noise(y, radius)
    theta = _add_noise(theta, radius)
    df = concat([Series(x),Series(y),Series(theta)], axis=1)
    df.columns = ["x", "y", "theta"]
    return df

def scale(col: DataFrame, _min: float, _max: float):
    return (col - _min) / (_max - _min)

def unscale(col: DataFrame, _min: float, _max: float):
    return col * (_max - _min) + _min