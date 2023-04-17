import numpy as np
import pandas as pd
import requests
from matplotlib import pyplot as plt
from matplotlib import rc
from IPython.core.display import HTML
from pyESN import ESN
styles = requests.get("https://raw.githubusercontent.com/Harvard-IACS/2018-CS109A/master/content/styles/cs109.css").text
HTML(styles)
rc('text', usetex=False)

data = open("apple.txt").read().split()
data = np.array(data).astype('float64')

n_reservoir= 500
sparsity=0.2
seed=23
spectral_radius = 1.5
noise = .0001

esn = ESN(n_inputs = 1,
      n_outputs = 1, 
      n_reservoir = n_reservoir,
      sparsity=sparsity,
      random_state=seed,
      spectral_radius = spectral_radius,
      noise=noise)

trainlen = 2800
future = 5
futureTotal = 200
pred_tot=np.zeros(futureTotal)

for i in range(0,futureTotal,future):
    pred_training = esn.fit(np.ones(trainlen),data[i:trainlen+i])
    prediction = esn.predict(np.ones(future))
    pred_tot[i:i+future] = prediction[:,0]
    print(prediction)


plt.figure(figsize=(16,8))
plt.plot(range(trainlen-365,trainlen+futureTotal),data[trainlen-365:trainlen+futureTotal],'b',label="Actual Data", alpha=0.3)
#plt.plot(range(0,trainlen),pred_training,'.g',  alpha=0.3)
plt.plot(range(trainlen,trainlen+futureTotal),pred_tot,'k',  alpha=0.8, label='ESN Prediction')

lo,hi = plt.ylim()
plt.plot([trainlen,trainlen],[lo+np.spacing(1),hi-np.spacing(1)],'k:', linewidth=4)

plt.title(r'Stock Predicitons: APPLE', fontsize=25)
plt.xlabel(r'Time (Days)', fontsize=20,labelpad=10)
plt.ylabel(r'Price ($)', fontsize=20,labelpad=10)
plt.legend(fontsize='xx-large', loc='best')
plt.show()