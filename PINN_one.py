"""在先前的基础上，加入反射边界条件，同时将公式换为不守恒形式"""
import deepxde as dde
import numpy as np
import math

# 解析解
def sol(X):
    x = X[:,0:1]
    y = X[:,1:2]
    t = X[:,2:3]
    hL = 5
    hR = 0.2
    g = 9.81
    c0 = math.sqrt(g * hR)
    c1 = math.sqrt(g * hL)
    c2 = 3.747613394
    c = 7.56902453
    u2 = 6.511648769
    L = 1000
    x1 = L / 2 - c1 * t
    x2 = (u2 - c2) * t + L / 2
    x3 = c * t + L / 2
    u = 0*x
    v = 0*x
    h = 0*x
    for i in range(x.shape[0]):
        if x[i]<x1[i]:
            h[i] = hL
            u[i] = 0
        elif (x1[i]<=x[i]<x2[i]):
            h[i] = (2*c1-(2*x[i]-L)/(2*t[i]))**2/(9*g)
            u[i] = (2*(x[i]+t[i]*math.sqrt(g*hL))-L)/(3*t[i])
        elif (x2[i]<=x[i]<x3[i]):
            h[i] = c2**2/g
            u[i] = u2
        elif x[i]>=x3[i]:
            h[i] = hR
            u[i] = 0
    Y = np.hstack((u,v,h))
    return Y

geom = dde.geometry.Rectangle((0,0),(1000,100))
timedomain = dde.geometry.TimeDomain(0,100)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# 控制方程
# 不带地形的二维非守恒方程
def ode_system_NCon_2D(X, Y):
    g=9.81
    x, y, t = X[:,0:1], X[:,1:2], X[:,2:3]
    u, v, h = Y[:, 0:1], Y[:, 1:2], Y[:, 2:3]
    dh_x = dde.grad.jacobian(h,X,i=0,j=0)
    dh_y = dde.grad.jacobian(h,X,i=0,j=1)
    dh_t = dde.grad.jacobian(h, X, i=0, j=2)

    du_x = dde.grad.jacobian(u,X,i=0,j=0)
    du_y = dde.grad.jacobian(u,X,i=0,j=1)
    du_t = dde.grad.jacobian(u, X, i=0, j=2)

    dv_x = dde.grad.jacobian(v,X,i=0,j=0)
    dv_y = dde.grad.jacobian(v,X,i=0,j=1)
    dv_t = dde.grad.jacobian(v, X, i=0, j=2)

    f_h = dh_t+u*dh_x+v*dh_y+h*(du_x+dv_y)
    f_u = du_t+u*du_x+v*du_y+g*dh_x
    f_v = dv_t+u*dv_x+v*dv_y+g*dh_y

    return [f_h,f_u,f_v]

#带地形的二维非守恒方程
def ode_system_NCon_2D_terrain(X, Y, Zb):
    g=9.81
    x, y, t = X[:,0:1], X[:,1:2], X[:,2:3]
    u, v, h = Y[:, 0:1], Y[:, 1:2], Y[:, 2:3]
    dh_x = dde.grad.jacobian(h,X,i=0,j=0)
    dh_y = dde.grad.jacobian(h,X,i=0,j=1)
    dh_t = dde.grad.jacobian(h, X, i=0, j=2)

    du_x = dde.grad.jacobian(u,X,i=0,j=0)
    du_y = dde.grad.jacobian(u,X,i=0,j=1)
    du_t = dde.grad.jacobian(u, X, i=0, j=2)

    dv_x = dde.grad.jacobian(v,X,i=0,j=0)
    dv_y = dde.grad.jacobian(v,X,i=0,j=1)
    dv_t = dde.grad.jacobian(v, X, i=0, j=2)

    dZb_x = dde.grad.jacobian(Zb,X,i=0,j=0)
    dZb_y = dde.grad.jacobian(Zb,X,i=0,j=1)

    f_h = dh_t+u*dh_x+v*dh_y+h*(du_x+dv_y)
    f_u = du_t+u*du_x+v*du_y+g*dh_x+g*dZb_x
    f_v = dv_t+u*dv_x+v*dv_y+g*dh_y+g*dZb_y

    return [f_h,f_u,f_v]

# 不带地形的二维守恒方程
def ode_system_Con_2D(X, Y):
    g=9.81
    x, y, t = X[:,0:1], X[:,1:2], X[:,2:3]
    u, v, h = Y[:, 0:1], Y[:, 1:2], Y[:, 2:3]
    dhu_x = dde.grad.jacobian(h*u,X,i=0,j=0)
    dhv_y = dde.grad.jacobian(h*v,X,i=0,j=1)
    dh_t = dde.grad.jacobian(h, X, i=0, j=2)

    du2_x = dde.grad.jacobian((h*u*u+g*h*h/2),X,i=0,j=0)
    dhuv_y = dde.grad.jacobian(h*u*v,X,i=0,j=1)
    dhu_t = dde.grad.jacobian(h*u, X, i=0, j=2)

    dhuv_x = dde.grad.jacobian(h*u*v,X,i=0,j=0)
    dv2_y = dde.grad.jacobian((h*v*v+g*h*h/2),X,i=0,j=1)
    dhv_t = dde.grad.jacobian(h*v, X, i=0, j=2)
    
    f_h = dh_t+dhu_x+dhv_y
    f_u = dhu_t+du2_x+v*dhuv_y
    f_v = dhv_t+dhuv_x+dhv_y

    return [f_h,f_u,f_v]

# 带地形的二维守恒方程
def ode_system_Con_2D_terrain(X, Y, Zb):
    g=9.81
    x, y, t = X[:,0:1], X[:,1:2], X[:,2:3]
    u, v, h = Y[:, 0:1], Y[:, 1:2], Y[:, 2:3]
    dhu_x = dde.grad.jacobian(h*u,X,i=0,j=0)
    dhv_y = dde.grad.jacobian(h*v,X,i=0,j=1)
    dh_t = dde.grad.jacobian(h, X, i=0, j=2)

    du2_x = dde.grad.jacobian((h*u*u+g*h*h/2),X,i=0,j=0)
    dhuv_y = dde.grad.jacobian(h*u*v,X,i=0,j=1)
    dhu_t = dde.grad.jacobian(h*u, X, i=0, j=2)

    dhuv_x = dde.grad.jacobian(h*u*v,X,i=0,j=0)
    dv2_y = dde.grad.jacobian((h*v*v+g*h*h/2),X,i=0,j=1)
    dhv_t = dde.grad.jacobian(h*v, X, i=0, j=2)

    dZb_x = dde.grad.jacobian(Zb,X,i=0,j=0)
    dZb_y = dde.grad.jacobian(Zb,X,i=0,j=1)

    f_h = dh_t+dhu_x+dhv_y
    f_u = dhu_t+du2_x+v*dhuv_y+g*h*dZb_x
    f_v = dhv_t+dhuv_x+dhv_y+g*h+dZb_y

    return [f_h,f_u,f_v]

#上游边界x=0
boundary_condition_u = dde.icbc.DirichletBC()
#下游边界x=L

#左侧边界y=0

#右侧边界y=H