import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit,learning_curve
from sklearn.pipeline import Pipeline
cancer=load_breast_cancer()
x=cancer.data
data=pd.DataFrame(x,columns=cancer.feature_names)
x=data.values
y=cancer.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
def plot_learning_curve(estimator,title,x,y,ylim=None,cv=None,n_jobs=1,train_size=np.linspace(.1,1.0,5)):
    plt.title(title)
    if ylim is not None:
        plt.ylim(min(ylim),max(ylim))
    plt.xlabel('Train examples')
    plt.ylabel('Score')
    train_size,train_scores,test_scores=learning_curve(estimator,x,y,n_jobs=n_jobs,cv=cv)
    train_score_mean=np.mean(train_scores,axis=1)
    train_score_std=np.std(train_scores,axis=1)
    test_score_mean=np.mean(test_scores,axis=1)
    test_score_std=np.std(test_scores,axis=1)
    plt.grid()
    plt.fill_between(train_size,train_score_mean-train_score_std,
                     train_score_mean+train_score_std,alpha=0.1,color='red')
    plt.fill_between(train_size,test_score_mean-test_score_std,
                     train_score_std+train_score_mean,alpha=0.1,color='green')
    plt.plot(train_size,train_score_mean,'o--',color='red',label='Training score')
    plt.plot(train_size,test_score_mean,'o--',color='green',label='Cross_validation score')
    plt.legend(loc='best')
    return plt

degrees=np.arange(1,11)
cv=ShuffleSplit(n_splits=10,test_size=10,random_state=0)

for degree in degrees:
    polynormianl=PolynomialFeatures(degree)
    mod1=LogisticRegression(penalty='l1')
    pipline=Pipeline([('polunormianl',polynormianl),('modl',mod1)])
    pipline.fit(x_train,y_train)
    plot_learning_curve(pipline,'logic Regression degree{}'.format(degree),x,y,cv=cv)
    plt.show()
    print(pipline.score(x_test,y_test))








