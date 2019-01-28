import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import time
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
def plot_learning_curve(estimator,title,x,y,ylim=None,cv=None,n_job=1,train_sizes=np.linspace(.1,1.0,5)):
    plt.title(title)
    if ylim is not None:
        plt.ylim(min(ylim),max(ylim))
    plt.xlabel('Traing examples')
    plt.ylabel('Score')
    train_sizes,train_scores,test_scores=learning_curve(estimator,x,y,cv=cv,n_jobs=n_job,train_sizes=train_sizes)
    train_scores_mean=np.mean(train_scores,axis=1)
    train_scores_std=np.std(train_scores,axis=1)
    test_scores_mean=np.mean(test_scores,axis=1)
    test_scores_std=np.std(test_scores,axis=1)
    plt.grid()
    plt.fill_between(train_sizes,train_scores_mean-train_scores_std,
                     train_scores_mean+train_scores_std,alpha=0.1,color='r')
    plt.fill_between(train_sizes,test_scores_mean-test_scores_std,
                     test_scores_mean+test_scores_std,alpha=0.1,color='g')
    plt.plot(train_sizes,train_scores_mean,'o--',color='r',label='Training score')
    plt.plot(train_sizes,test_scores_mean,'o-',color='g',label='Cross-validation score')
    plt.legend(loc='best')
    return plt


def plotnomial_model(degree=1):
    plotnomial_model=PolynomialFeatures(degree=degree,include_bias=False)
    linear_regression=LinearRegression(normalize=False)
    pipeline=Pipeline([('plotnomial_model',plotnomial_model),('linear_regression',linear_regression)])
    return pipeline

def main():
    '''
    字段说明
    1、CRM       城镇人均犯罪率
    2、ZN        住宅用地所占比例，25000英寸
    3、INDUS     城镇中非商业用地的所占比例
    4、CHAS      查尔斯河虚拟变量
    5、NOX       环保指标
    6、RM        每栋住宅的房间数
    7、AGE       1940年以前建成的自主单位比例
    8、DIS       距离5个波士顿就业中心的加权距离
    9、RAD       距离告诉公路的便利指标
    10、TAX      每一万美圆的不动产税率
    11、RTRATIO  城镇中教师学生比例
    12、B        城镇中黑人的比例
    13、LSTAT    地区有多少百分比的房东属于低收入阶层
    14、MEDV     自主房屋房价中位数
    '''
    boston=load_boston()
    x=boston.data
    y=boston.target
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
    model=LinearRegression()
    start=time.clock()
    model.fit(x_train,y_train)
    train_score=model.score(x_train,y_train)
    cv_score=model.score(x_test,y_test)
    # print('elsep:{0:.6f};train_data:{1:.6f};test_data:{2:.6f}'.format(time.clock()-start,train_score,cv_score))
    cv=ShuffleSplit(n_splits=10,test_size=10,random_state=0)
    plt.figure(figsize=(18,4),dpi=200)
    title='Learning Curvers(degree={0})'
    degrees=[1,2,3]

    start=time.clock()
    plt.figure(figsize=(18,4),dpi=200)
    for i in range(len(degrees)):
        plt.subplot(1,3,i+1)
        plot_learning_curve(plotnomial_model(degrees[i]),title.format(degrees[i]),x,y,ylim=(0.01,1.01),cv=cv)

    print('elaspe:{0:.6f}'.format(time.clock()-start))

if __name__ == '__main__':
    main()
    plt.show()











