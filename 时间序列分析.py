import numpy as np
import warnings
import itertools
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
data=sm.datasets.co2.load_pandas()
y=data.data
y=y['co2'].resample('MS').mean()
y=y.fillna(y.bfill())
y.plot()
# plt.show()
#参数确定
p=d=q=range(0,2)
pdq=list(itertools.product(p,d,q))
seasonal_pdq=[(x[0],x[1],x[2],12) for x in list(itertools.product(p,d,q))]
# print(seasonal_pdq)
print('Example of parameter combinations for Seasonal ARIMA...')
print('SARIMAR:{},x{}'.format(pdq[1],seasonal_pdq[1]))
print('SARIMAR:{},x{}'.format(pdq[1],seasonal_pdq[2]))
print('SARIMAR:{},x{}'.format(pdq[2],seasonal_pdq[3]))
print('SARIMAR:{},x{}'.format(pdq[2],seasonal_pdq[4]))

#使用AIC确定参数
warnings.filterwarnings('ignore')
p=d=q=None
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod=sm.tsa.statespace.SARIMAX(y,order=param,seasonal_order=param_seasonal,enforce_invertibility=False,enforce_stationarity=False)
            result=mod.fit()

            print('ARIMA{}x{}-AIC:{}'.format(param,param_seasonal,result.aic))
        except:
            continue

mod=sm.tsa.statespace.SARIMAX(y,order=(1,1,1),seasonal_order=(1,1,1,12),enforce_stationarity=False,enforce_invertibility=False)
result=mod.fit()
result.plot_diagnostics()
plt.show(figsize=(15,12))
pred=result.get_prediction(start=pd.to_datetime('1998-01-01'),dynamic=True,full_result=True)
pred_ic=pred.conf_int()
print(pred.predicted_mean)
ax=y['1990':].plot(label='observed')
pred.predicted_mean.plot(ax=ax,label='One-step,ahed Forecast',alpha=.7)
ax.fill_between(pred_ic.index,pred_ic.iloc[:,0],pred_ic.iloc[:,1],color='k',alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('co2 levels')
plt.legend() #显示图例

plt.show()
# y=np.random.random_integers(300,400,(300,))
# y=pd.Series(y,index=pd.date_range('1958-1-1',periods=300,freq='m'))
# y.plot()
# plt.show()
# print(y)
