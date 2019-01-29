import matplotlib.pyplot as plt
import numpy as np
import itertools

def plt_scatter():
    n=1024
    x=np.random.normal(0,1,n)
    y=np.random.normal(0,1,n)
    t=np.arctan2(y,x)
    plt.subplot(1,2,1)
    plt.scatter(x,y,s=75,c=t,alpha=.5)
    plt.xlim(-1.5,1.5)
    plt.xticks()
    plt.ylim(-1.5,1.5)
    plt.yticks()

def plt_fill_between():
    n=256
    x=np.linspace(-np.pi,np.pi,endpoint=True)
    y=np.sin(2*x)
    plt.subplot(1,2,2)
    plt.plot(x,y+1,color='blue',alpha=1.00)
    plt.fill_between(x,1+y,color='blue',alpha=.25)
    plt.plot(x,y-1,color='blue',alpha=1)
    plt.fill_between(x,-1,y-1,(y-1)>-1,color='blue',alpha=.25)
    plt.fill_between(x,-1,y-1,(y-1)<-1,color='red',alpha=.25)
    plt.xlim(-np.pi,np.pi)
    plt.xticks(())
    plt.ylim(-2.5,2.5)
    plt.yticks(())

def plt_bar():
    n=12
    x=np.arange(n)
    plt.subplot(1,2,1)
    y1=(1-x/float(n))*np.random.uniform(0.5,1.0,n)
    y2=(1-x/float(n))*np.random.uniform(0.5,1.0,n)

    plt.subplot(1,2,1)
    plt.bar(x,y1,facecolor='#9999ff',edgecolor="white")
    plt.bar(x,-y2,facecolor='#ff9999',edgecolor='white')
    for i,j in zip(x,y1):
        plt.text(i,j+0.05,'%.2f'%j,ha='center',va='bottom')

    for i,j in zip(x,y2):
        plt.text(i,-j-0.05,'%.2f'%j,ha='center',va='top')
        # print()

    plt.xlim(-.5,n)
    plt.xticks()
    plt.ylim(-1.25,1.25)
    plt.yticks()

def plt_imshow():
    def f(x,y):
        return (1-x/2+x**5+y**3)*np.exp(-x**2-y**2)
    plt.subplot(1,2,1)
    n=10
    x=np.linspace(-3,3,4*n)
    y=np.linspace(-3,3,3*n)
    X,Y=np.meshgrid(x,y)
    plt.imshow(f(X,Y),cmap='hot',origin='low')
    plt.colorbar(shrink=.83)
    plt.xticks()
    plt.yticks()
    plt.show()

def plt_contour():
    def f(x,y):
        return (1-x/2+x**5+y**3)*np.exp(-x**2-y**2)
    n=256
    x=np.linspace(-3,3,n)
    y=np.linspace(-3,3,n)
    X,Y=np.meshgrid(x,y)
    plt.contourf(X,Y,f(X,Y),8,alpha=.75,cmap=plt.cm.hot)
    c=plt.contour(X,Y,f(X,Y),8,color='black',linewidth=.5)
    plt.clabel(c,inline=1,fontsize=10)
    plt.xticks()
    plt.yticks()
    plt.show()

def plt_pie():
    '''
    饼图
    '''
    n=20
    z=np.ones(n)
    z[-1]*=2
    # print(z)
    plt.pie(z,explode=z*.0000001,colors=['%f'%(i/float(n)) for i in range(n)])
    plt.axis('equal')
    plt.xticks()
    plt.yticks()
    plt.show()

def plt_polar():
    ax=plt.subplot(polar=True)
    N=20
    theta=np.arange(0.0,2*np.pi,2*np.pi/N)
    radii=10*np.random.rand(N)
    width=np.pi/4*np.random.rand(N)
    bars=plt.bar(theta,radii,width=width,bottom=0.0,)
    for r,bar in zip(radii,bars):
        bar.set_facecolor(plt.cm.jet(r/10.))
        bar.set_alpha(0.5)
    plt.show()

def plt_radar():
    #标签
    label=np.array(['3℃','5℃','6℃','1℃','3℃','3℃','2℃'])
    #数据
    np.random.seed(1)
    datas=np.random.randint(1,8,size=label.shape[0])
    # print(datas)
    angles=np.linspace(0,2*np.pi,datas.shape[0],endpoint=False)
    # print(angles)
    data=np.concatenate((datas,[datas[0]]))
    angle=np.concatenate((angles,[angles[0]]))
    plt.figure()
    ax=plt.subplot(polar=True)
    ax.plot(angle,data,"ro-",linewidth=2)
    ax.set_thetagrids(angles*180/np.pi,label)
    ax.grid(True)

    plt.show()
if __name__ == '__main__':

    # plt.figure(figsize=(16, 6))
    # plt_scatter()
    # plt_fill_between()
    # plt.show()
    #
    # plt_bar()
    # plt.show()
    # plt_imshow()
    # plt_contour()
    # plt_pie()
    # plt_polar()
    plt_radar()