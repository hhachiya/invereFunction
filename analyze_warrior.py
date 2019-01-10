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
	fig = pkl.load(fp)

# plot image	
fig.show()
plt.savefig('analysis_cluster.png')

# top 5 words for each cluster
input('Enterを押すと, 各クラスター中心に関連する語top5を表示します:')
for c in range(len(top_similar_words)):
	print("クラスター{}: {}".format(c,top_similar_words[c][:5]))
print("\n")

for c in range(len(warrior_names_by_cluster)):
	input('Enterを押すと, クラスター{}に属する武将一覧を表示します:'.format(c))
	print(warrior_names_by_cluster[c])
	print("\n")

input('Enterを押すと, 終了します')
