# coding: utf-8

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np

data = np.array([1., 2., 2., 2., 3., 3., 4., 4., 5., 6., 7., 10.]).reshape([-1,1])

scalers = [
    { 'scaler' : None, \
	  'title' : 'Original' },
    { 'scaler' : preprocessing.MinMaxScaler(), \
      'title' : 'MinMaxScaler' },
    { 'scaler' : preprocessing.StandardScaler(with_mean=False, with_std=False), \
      'title' : 'StandardScaler (mean=false, std=false)' },
    { 'scaler' : preprocessing.StandardScaler(with_mean=True, with_std=False), \
      'title' : 'StandardScaler (mean=true, std=false)' },
    { 'scaler' : preprocessing.StandardScaler(with_mean=False, with_std=True), \
      'title' : 'StandardScaler (mean=false, std=true)' },
    { 'scaler' : preprocessing.StandardScaler(with_mean=True, with_std=True), \
      'title' : 'StandardScaler (mean=true, std=true)' }
]
n_scalers = len(scalers)

def draw_plot(subplot, scaler, title, data):
    plt.subplot(subplot[0], subplot[1], subplot[2])
    plt.title(title)
    plt.hist(data, alpha=0.2)
    text_x = min(data) + ( ( max(data) - min(data) ) / 2.0 )
    text_y = 2
    text = str( np.round(data, 3).tolist() )
    plt.text(text_x, text_y, text, ha='center', va='center', size='small', \
            bbox={'facecolor':'red', 'alpha':0.5, 'pad':10, 'alpha':0.2})

plt.figure(figsize=(10, 13))
for i in range(n_scalers):
    subplot = (n_scalers, 1, i+1)
    scaler = scalers[i]['scaler']
    title = scalers[i]['title']
    if scaler:
        transformed_data = scaler.fit_transform(data)
    else:
        transformed_data = data
    draw_plot(subplot, scaler, title, transformed_data)
plt.tight_layout()
plt.savefig('scaler.png')

