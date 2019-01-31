import matplotlib.pyplot as plt
import numpy as np
#成本函数
def f_1(x):
    return -np.log(x)

def f_0(x):
    return -np.log(1-x)

X=np.linspace(.01,.99,100)
f=[f_1,f_0]
title=["y=1:$-log(h_\\theta(x))$","y=0:$-log(1-h_\\theta(x))$"]
plt.figure(figsize=(12,4),dpi=144)
for i in range(len(f)):
    plt.subplot(1,2,i+1)
    plt.title(title[i])
    plt.xlabel("$h_\\theta(x)$")
    plt.ylabel("$COST(h_\\theta(x),y)$")
    plt.plot(X,f[i](X),'r-')
plt.show()
#L1/L2范数
def L1(x):
    return  1-abs(x)

def L2(x):
    return np.square(1-np.power(x+2,2))

def format_spines(title):
    ax=plt.gca()  #电表当前坐标轴
    ax.spines['right'].set_color('none')  #隐藏坐标轴
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')  #设置刻度显示位置
    ax.spines['bottom'].set_position(('data',0))   #设置下方坐标轴位置
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data',0))   #设置左侧坐标轴位置

    plt.title(title)
    plt.xlim(-4,4)
    plt.ylim(-4,4)

plt.figure(figsize=(18,4),dpi=144)
x=np.linspace(-1,1,100)
cx=np.linspace(-3,-1,100)

plt.subplot(1,2,1)
format_spines('L1 norm')
plt.plot(x,L1(x),'r-',x,-L1(x),'r-')
plt.plot(cx,contour(20,cx), 'r--',cx,contour(15, cx,), 'r--',cx,contour(10, cx), 'r--')


# plt.subplot(1,2,2)
# format_spines('L2 norm')
# plt.plot
