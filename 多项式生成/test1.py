import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from matplotlib.figure import SubplotParams

n_dot=200
x=np.linspace(-2*np.pi,2*np.pi,n_dot)
y=np.sin(x)+0.2*np.random.rand(n_dot)-0.1
x=x.reshape(-1,1)
y=y.reshape(-1,1)
plt.plot(x,y)
plt.show()

def polynoimal_model(degree=1):
    polynoimal_features=PolynomialFeatures(degree=degree,include_bias=False)
    linear_regression=LinearRegression(normalize=None)
    pipline=Pipeline([('polynoimal_features',polynoimal_features),
                      ('Linear_regression',linear_regression)])
    return pipline

degress=[2,3,5,10]
result=[]
for d in degress:
    model=polynoimal_model(degree=d)
    model.fit(x,y)
    train_score=model.score(x,y)
    mse=mean_squared_error(y,model.predict(x))
    result.append({'model':model,'degress':d,'score':train_score,'mse':mse})
for r in result:
    print('drgress:{},train score:{};mean squared error:{}'.format(r['degress'],r['score'],r['mse']))

plt.figure(figsize=(12,6),dpi=200,subplotpars=SubplotParams(hspace=0.3))
for i,r in enumerate(result):
    fig=plt.subplot(2,2,i+1)
    plt.xlim(-8,8)
    plt.title('LinearRegression degress={}'.format(r['degress']))
    plt.scatter(x,y,s=5,c='b',alpha=0.5)
    plt.plot(x,r["model"].predict(x),'r-')
    plt.show()
