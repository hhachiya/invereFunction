import pdb
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle as pkl
import os
import matplotlib.pylab as plt
from gensim.models import word2vec
from sklearn.cluster import KMeans

batch_size = 100
train_count = 0
test_count = 0
ite = 50000
src_vec_length = 2
hidden_unit_num = 500
tar_vec_length = 100
method = 1 #1:w2v,2:tfidf
train_per = 0.9
nCluster = 10

#===========================
# レイヤーの関数
# fc layer
def fc_relu(inputs, w, b):
	fc = tf.matmul(inputs, w) + b
	fc = tf.nn.relu(fc)
	return fc
	
# fc layer
def fc(inputs, w, b):
	fc = tf.matmul(inputs, w) + b
	return fc

def weight_variable(name,shape):
    return tf.get_variable(name,shape,initializer=tf.random_normal_initializer(stddev=0.1))

def bias_variable(name,shape):
    return tf.get_variable(name,shape,initializer=tf.constant_initializer(0.1))
#===========================

#===========================
# tensorflowで用いるデータ群作成
input_data = tf.placeholder(shape=(None,None),dtype = tf.float32,name='input_data')
target_data = tf.placeholder(shape=(None,None),dtype = tf.float32,name='target_data')

# 線形回帰で必要なW,Bを作成
W1 = weight_variable('weight1',[src_vec_length,hidden_unit_num])
B1 = weight_variable('bias1',[hidden_unit_num])
W2 = weight_variable('weight2',[hidden_unit_num,hidden_unit_num])
B2 = weight_variable('bias2',[hidden_unit_num])
W3 = weight_variable('weight3',[hidden_unit_num,tar_vec_length])
B3 = weight_variable('bias3',[tar_vec_length])

# 活性化関数を指定してfc層を生成。
fc1 = fc_relu(input_data, W1, B1)
fc2 = fc_relu(fc1, W2, B2)
fc_out = fc(fc2,W3,B3)
#===========================

#===========================
# loss関数(平均二乗誤差)
loss = tf.reduce_mean(tf.abs(target_data - fc_out))

# optimizerの設定
train_optimaizer = tf.train.AdamOptimizer(0.01).minimize(loss)

# 初期化
sess = tf.Session()
sess.run(tf.global_variables_initializer())
#===========================

#===========================
# 使用する手法ごとにデータパスを作成する。
if method == 1:
    src_dataPath = 'warrior-data-TSNE/w2v/data_label.pickle'
    tar_dataPath = 'warrior-data-pkl/w2v'

elif method == 2:
    src_dataPath = 'warrior-data-TSNE/tfidf/data_label.pickle'
    tar_dataPath = 'warrior-data-pkl/tfidf'

clusterOutPath = 'warrior-data-cluster/analysis.pickle'
warriorNamePath = 'warrior-data-txt/warrior_name.txt'
modelPath = 'wikipedia-data-model/word2vec.model'
#===========================


#===========================
# T-SNEで２次元に削減したデータと正解ラベルのペアをpickleファイルから読み込む
with open(src_dataPath,'rb') as fp:
	src_data = pkl.load(fp)
	src_label = pkl.load(fp)

# train用、test用で分割する。
randInd_all = np.random.permutation(src_data.shape[0])
src_train_data = src_data[randInd_all[:int(src_data.shape[0]*train_per)]]
src_test_data = src_data[randInd_all[int(src_data.shape[0]*train_per):]]
randInd_train = np.random.permutation(src_train_data.shape[0])
#===========================


#===========================
# word2vecの100次元のベクトルデータを読み込む
# ファイル名（カテゴリに対応）をまとめているテキストからファイル名を読み出す
f = open('file_name_warrior.txt','r')
# ファイル名がすべて接続されて居るので区切り文字を指定してファイル名ごとに分ける
lines = f.readlines()
fn = [line.strip() for line in lines]
flag = 0
for fInd in np.arange(len(fn)):
	tar_fulldataPath = os.path.join(tar_dataPath,'{}.pickle'.format(fn[fInd]))
	with open(tar_fulldataPath,'rb') as fp:
		tmpX = pkl.load(fp)
		tmpY = np.ones(tmpX.shape[0])*fInd

		if flag == 0:
			tar_data = tmpX
			tar_label = tmpY
			flag = 1
		else:
			tar_data = np.vstack([tar_data,tmpX])
			tar_label = np.hstack([tar_label,tmpY])
			

# train用、test用で分割する。
# 標準偏差が1になるように正規化
tar_data = tar_data/np.tile(np.std(tar_data,axis=0),[tar_data.shape[0],1])
tar_train_data = tar_data[randInd_all[:int(tar_data.shape[0]*train_per)]]
tar_test_data = tar_data[randInd_all[int(tar_data.shape[0]*train_per):]]
#===========================

#===========================
# batch作成
def next_batch_train(count,Ind):
    #epochが終了しているならばIndexを作り直す
    if(src_train_data.shape[0] < (batch_size*(count+1))):
        Ind = np.random.permutation(src_train_data.shape[0])
        count = 0
    
    # batch_size分だけ取り出して、カウントを増やす
    batch_src = src_train_data[Ind[batch_size*count:batch_size*(count+1)]]
    batch_tar = tar_train_data[Ind[batch_size*count:batch_size*(count+1)]]
    
    # 辞書型にして返す
    return {
        input_data:batch_src,
        target_data:batch_tar,
    },count+1,Ind

#===========================

#===========================
# 過程のsave用に必要なクラスを宣言
#saver=tf.train.Saver()
# 学習の開始
for step in np.arange(ite+1):
    fd,train_count,randInd_train = next_batch_train(train_count,randInd_train)
    _, step_loss = sess.run([train_optimaizer,loss],feed_dict=fd)
    if step%1 == 0:
        print('step:{},loss:{}'.format(step,str(step_loss)))
        #saver.save(sess,'inverse_model/model.ckpt',global_step=step)

    if step%100 ==0:
        step_loss,predicted = sess.run([loss,fc_out],feed_dict={input_data:src_test_data, target_data: tar_test_data})
        print('---------test_sequence---------')
        print('loss:{}'.format(str(step_loss)))
        print('input ->:{}'.format(fd[input_data][0]))
        print('predicted ->:{}'.format(str(predicted[0])))
        print('gt ->:{}'.format(str(tar_test_data[0])))
        print('-------------------------------')
#===========================

model = word2vec.Word2Vec.load(modelPath)

'''
#===========================
# check specified points words
target_point = np.array([[-5,50],[5,50],[50,5],[50,-5],[5,-50],[-5,-50],[-50,-5],[-50,5]])
target_predict = sess.run(fc_out,feed_dict={input_data:target_point,target_data:np.zeros(shape=(8,100))})

for p in range(len(target_point)):
	print(target_point[p])
	print(model.wv.similar_by_vector(target_predict[p]))
#===========================
'''

print("----------------------------")

#===========================
# check clustered points words
kmeans = KMeans(n_clusters=nCluster, random_state=0).fit(src_data)

nSample = 50
top_similar_words = []
for c in range(nCluster):
	target_point = np.random.normal(size=[nSample,2])*np.tile(np.std(src_data[kmeans.labels_==c],axis=0)*0.1,[nSample,1])+kmeans.cluster_centers_[c]
	target_predict = sess.run(fc_out,feed_dict={input_data:target_point,target_data:np.zeros(shape=(nSample,tar_vec_length))})

	flag = True
	for s in range(nSample):
		similar_words = model.wv.similar_by_vector(target_predict[s])

		if flag:
			similar_words_dict = dict(similar_words)
			flag = False
		else:
			for key_score in similar_words:
				if key_score[0] in similar_words_dict.keys():
					similar_words_dict[key_score[0]] += key_score[1]
				else:
					similar_words_dict[key_score[0]] = key_score[1]

	top_similar_words.append([k for k, v in sorted(similar_words_dict.items(), key=lambda x:x[1])[::-1]])
#===========================

#===========================
# warrior names
with open(warriorNamePath,'rt') as fp:
	warrior_names = fp.readlines()
	warrior_names = np.array([name.strip() for name in warrior_names])

warrior_names_by_cluster = [warrior_names[kmeans.labels_ == c] for c in range(nCluster)]
#===========================

#----------------------
lcmap =['#000000', '#FF99FF', '#8000FF','#0000FF', '#0080FF', '#58FAF4','#00FF00', '#FFFF00', '#FF8000','#FF0000']

fig = plt.figure()

for c in range(nCluster):
	plt.plot(src_data[kmeans.labels_==c,0],src_data[kmeans.labels_==c,1],'o',markerfacecolor=lcmap[c])
	plt.text(kmeans.cluster_centers_[c,0],kmeans.cluster_centers_[c,1],str(c),fontsize=15,bbox=dict(facecolor='white', alpha=0.5))
#----------------------

with open(clusterOutPath,'wb') as fp:
	pkl.dump(src_data,fp)
	pkl.dump(tar_data,fp)
	pkl.dump(top_similar_words,fp)
	pkl.dump(kmeans,fp)
	pkl.dump(warrior_names,fp)
	pkl.dump(warrior_names_by_cluster,fp)
	pkl.dump(fig,fp)

