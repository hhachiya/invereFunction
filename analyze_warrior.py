import pdb
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle as pkl
import os
import matplotlib.pylab as plt
from sklearn.cluster import KMeans

clusterOutPath = 'warrior-data-cluster/analysis.pickle'

with open(clusterOutPath,'rb') as fp:
	src_data = pkl.load(fp)
	tar_data = pkl.load(fp)
	top_similar_words = pkl.load(fp)
	kmeans = pkl.load(fp)
	warrior_names = pkl.load(fp)
	warrior_names_by_cluster = pkl.load(fp)
	pdb.set_trace()
	fig = pkl.load(fp)
	
fig.show()

pdb.set_trace()
