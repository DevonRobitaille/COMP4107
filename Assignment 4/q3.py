#Code inspired/taken from https://www.kaggle.com/asparago/unsupervised-learning-with-som
#Fitted to suit needs of assignment


import numpy as np 
import pandas as pd 
import seaborn as sns
from imageio import imwrite
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data
%matplotlib inline

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
from k_means import cluster

from sklearn.preprocessing import StandardScaler
from PIL import Image, ImageChops

import SimpSOM as sps

np.random.seed(0)

#reading the data
train_data = pd.read_csv('../input/train.csv')
train_data = train_data[train_data['label']==1]
train_data = train_data.sample(n=500, random_state=0)

labels = train_data['label']
train_data = train_data.drop("label",axis=1)

sns.displot(labels.values,bins=np.arange(-0.5,10.5,1))

trainSt = StandardScaler().fit_transform(train_data.values)

net = sps.somNet(28, 28, trainSt, PBC=True, PCI=True)

#Given 0.5 learning rate and 1000 epochs for sake of testing
net.train(0.5, 1000)

net.diff_graph(show=True,printout=True)

