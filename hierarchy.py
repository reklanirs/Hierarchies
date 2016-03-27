#!/usr/bin/python2.7
# -*- coding: utf-8 -*- import requests
import os
import re
import sys
import math
import random
import requests
import time
import json
import lxml
import demjson
from random import choice
from bs4 import BeautifulSoup
from bs4 import UnicodeDammit
from lxml import etree
import jieba
import gensim, logging
from sklearn import cluster
import numpy as np
from numbapro import vectorize
from numbapro import cuda
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

reload(sys)

all_word = {}
childnum = {}
model = gensim.models.Word2Vec.load('model/m100.w2v')
print 'Model loaded'
triple = []
tmptriple = ['','','']
alltriple = []

colorsbk = ['#FFFAFA','#F8F8FF','#F5F5F5','#DCDCDC','#FFFAF0','#FDF5E6','#FAF0E6','#FAEBD7','#FFEFD5','#FFEBCD','#FFE4C4','#FFDAB9','#FFDEAD','#FFE4B5','#FFF8DC','#FFFFF0','#FFFACD','#FFF5EE','#F0FFF0','#F5FFFA','#F0FFFF','#F0F8FF','#E6E6FA','#FFF0F5','#FFE4E1','#FFFFFF','#000000','#2F4F4F','#696969','#708090','#778899','#BEBEBE','#D3D3D3','#191970','#000080','#6495ED','#483D8B','#6A5ACD','#7B68EE','#8470FF','#0000CD','#4169E1','#0000FF','#1E90FF','#00BFFF','#87CEEB','#87CEFA','#4682B4','#B0C4DE','#ADD8E6','#B0E0E6','#AFEEEE','#00CED1','#48D1CC','#40E0D0','#00FFFF','#E0FFFF','#5F9EA0','#66CDAA','#7FFFD4','#006400','#556B2F','#8FBC8F','#2E8B57','#3CB371','#20B2AA','#98FB98','#00FF7F','#7CFC00','#00FF00','#7FFF00','#00FA9A','#ADFF2F','#32CD32','#9ACD32','#228B22','#6B8E23','#BDB76B','#EEE8AA','#FAFAD2','#FFFFE0','#FFFF00','#FFD700','#EEDD82','#DAA520','#B8860B','#BC8F8F','#CD5C5C','#8B4513','#A0522D','#CD853F','#DEB887','#F5F5DC','#F5DEB3','#F4A460','#D2B48C','#D2691E','#B22222','#A52A2A','#E9967A','#FA8072','#FFA07A','#FFA500','#FF8C00','#FF7F50','#F08080','#FF6347','#FF4500','#FF0000','#FF69B4','#FF1493','#FFC0CB','#FFB6C1','#DB7093','#B03060','#C71585','#D02090','#FF00FF','#EE82EE','#DDA0DD','#DA70D6','#BA55D3','#9932CC','#9400D3','#8A2BE2','#A020F0','#9370DB','#D8BFD8','#FFFAFA','#EEE9E9','#CDC9C9','#8B8989','#FFF5EE','#EEE5DE','#CDC5BF','#8B8682','#FFEFDB','#EEDFCC','#CDC0B0','#8B8378','#FFE4C4','#EED5B7','#CDB79E','#8B7D6B','#FFDAB9','#EECBAD','#CDAF95','#8B7765','#FFDEAD','#EECFA1','#CDB38B','#8B795E','#FFFACD','#EEE9BF','#CDC9A5','#8B8970','#FFF8DC','#EEE8CD','#CDC8B1','#8B8878','#FFFFF0','#EEEEE0','#CDCDC1','#8B8B83','#F0FFF0','#E0EEE0','#C1CDC1','#838B83','#FFF0F5','#EEE0E5','#CDC1C5','#8B8386','#FFE4E1','#EED5D2','#CDB7B5','#8B7D7B','#F0FFFF','#E0EEEE','#C1CDCD','#838B8B','#836FFF','#7A67EE','#6959CD','#473C8B','#4876FF','#436EEE','#3A5FCD','#27408B','#0000FF','#0000EE','#0000CD','#00008B','#1E90FF','#1C86EE','#1874CD','#104E8B','#63B8FF','#5CACEE','#4F94CD','#36648B','#00BFFF','#00B2EE','#009ACD','#00688B','#87CEFF','#7EC0EE','#6CA6CD','#4A708B','#B0E2FF','#A4D3EE','#8DB6CD','#607B8B','#C6E2FF','#B9D3EE','#9FB6CD','#6C7B8B','#CAE1FF','#BCD2EE','#A2B5CD','#6E7B8B','#BFEFFF','#B2DFEE','#9AC0CD','#68838B','#E0FFFF','#D1EEEE','#B4CDCD','#7A8B8B']
colors = ['#E6E6FA', '#00BFFF', '#98FB98', '#8B4513', '#FF1493', '#FFF5EE', '#CDC9A5', '#F0FFFF', '#4F94CD']

kcluster = 5
kms = cluster.KMeans(n_clusters = kcluster)
cluster_centers_ = []
labels_ = []
phis = []
finalcosts = []

wrongdict = {}
rightdict = {}
rpos = {}
uneg = {}
upos = {}
rneg = {}
sumnum = {}

sysstdout = sys.stdout
fout = open('out.txt','w')
def writeln(s):
	print s
	fout.write(str(s) + '\n')
	fout.flush()

def write(s):
	print s,
	fout.write(str(s) + ' ')
	fout.flush()

def affirm(_s = 'show PCA Graph'):
	s = 'Need ' + _s + '? y to show and anything else to continue'
	writeln(s)
	s = raw_input().strip()
	if s == 'y':
		return True
	return False

class node(object):
	"""docstring for node"""
	def __init__(self, _value = [], _keylen = 1, _level = 0):
		self.value = _value
		self.child = {}
		self.keylen = _keylen
		self.level = _level
	def traverse(self, t = 0, flag = False):
		if self.keylen == 0:
			tmp = len(self.value)
			if tmp not in childnum:
				childnum[tmp] = 1
			else:
				childnum[tmp] += 1

		if self.level == 2 or self.level == 3:
			tmptriple[self.level - 2] = self.value
		if self.level == 5:
			tmptriple[2] = self.value
			triple.append(tmptriple[:])

		if flag:
			for i in range(t):
				write('\t')
			for i in self.value:
				write(i.encode('utf-8'))
			writeln('')

		for i,j in self.child.items():
			j.traverse(t + 1, flag)
		pass

root = node(['ROOT'],1)

def readfile(_file):
	for i in open(_file):
		yield i.strip().decode('utf-8').strip()

def dis2(x,y):
	return dis(x,y)**2
	pass

def dis(x,y):
	return np.linalg.norm(x-y)
	pass

def makeTree():
	global root
	#l12
	for i in readfile('CilinE/l12.txt'):
		# writeln(i.encode('utf-8')
		tmp = i.split()
		key = tmp[0]
		value = tmp[1:]
		all_word[key] = value
		if len(key) == 1:
			tmp = node(value,1,1)
			root.child[key] = tmp
			pass
		elif len(key) == 2:
			tmp = node(value,2,2)
			root.child[key[0]].child[key[1]] = tmp
			pass
		else:
			writeln('Error '+i)
			pass


	#l3
	for i in readfile('CilinE/l3.txt'):
		tmp = i.split()
		key = tmp[0][:-1]
		value = tmp[1:]
		all_word[key] = value
		tmp = node(value,1,3)
		root.child[key[0]].child[key[1]].child[key[2:]] = tmp

	#l45
	for i in readfile('CilinE/l45.txt'):
		tmp = i.split()
		key = tmp[0][:-1]
		symbol = tmp[0][-1]
		value = tmp[1:]
		all_word[key] = value

		l4 = key[4]
		l5 = key[5:]
		if l4 not in root.child[key[0]].child[key[1]].child[key[2:4]].child:
			tmp = node([l4],2,4)
			root.child[key[0]].child[key[1]].child[key[2:4]].child[l4] = tmp
		# tmp = node([symbol + i for i in value], 0, 5)
		tmp = node(value, 0, 5)
		root.child[key[0]].child[key[1]].child[key[2:4]].child[l4].child[l5] = tmp

	writeln('MakeTree End')
	pass


def cost(phi,x,y):
	return dis2(np.dot(x,phi), y)
	pass

def costAll(phi,X,Y):
	return dis2(np.dot(X,phi), Y)/len(X)
	pass


def calcEulerDisInClusters(x,y):
	vector_x = model[x.encode('utf-8')]
	vector_y = model[y.encode('utf-8')]
	difference = vector_y - vector_x
	mindis = 1e10
	mink = -1
	for i in xrange(kcluster):
		center = cluster_centers_[i]
		tmpdis = dis(difference, center)
		if tmpdis < mindis:
			mindis = tmpdis
			mink = i

	return cost(phis[mink], vector_x, vector_y),mink


def calcPhiSGD(X,Y):
	D = len(X[0])
	phi = np.random.sample((D,D))
	eps = 1e-3
	r = 0.1
	rt = 0.999
	loopnum = 5000
	for step in xrange(loopnum):
		if step%10 == 0:
			writeln('\tStep %d %f'%(step,costAll(phi, X, Y)))

		tmp = random.choice(xrange(len(X)))
		x = X[tmp]
		y = Y[tmp]
		Edict = {}
		for v in xrange(D):
			tmp = 0.0
			for k in xrange(D):
				tmp += x[k]*phi[k][v]
			Edict[v] = tmp

		for u in xrange(D):
			for v in xrange(D):
				jt = 2.0 * x[u] * (Edict[v] - y[v])
				phi[u][v] -= r * jt

		r *= rt
		pass
	cost = costAll(phi, X, Y)
	writeln('\tStep %d %f'%(loopnum,cost))
	return phi,cost






def ForTest(related, unrelated, thresholdrate = 1.2, train = [], output = False):
	relatedpos = 0
	relatedneg = 0
	unrelatedpos = 0
	unrelatedneg = 0

	rpl = []
	rnl = []
	upl = []
	unl = []

	rmin = 1e8
	rmax = 0
	for i in xrange(kcluster):
		wrongdict[i] = 0
		rightdict[i] = 0
		rpos[i] = 0
		uneg[i] = 0
		upos[i] = 0
		rneg[i] = 0
		rpl.append([])
		rnl.append([])
		upl.append([])
		unl.append([])

	for x,y in related:
		d,clusternum = calcEulerDisInClusters(x,y)
		rmin = min(rmin, d)
		rmax = max(rmax, d)

		if d < thresholdrate * finalcosts[clusternum]:
			relatedpos += 1
			rightdict[clusternum] += 1
			rpos[clusternum]+=1
			rpl[clusternum].append((x,y))
		else:
			relatedneg += 1
			wrongdict[clusternum] += 1
			rneg[clusternum] += 1
			rnl[clusternum].append((x,y))

	umin = 1e8
	umax = 0
	for x,y in unrelated:
		d,clusternum = calcEulerDisInClusters(x,y)
		umin = min(umin, d)
		umax = max(umax, d)

		if d < thresholdrate * finalcosts[clusternum]:
			unrelatedpos += 1
			wrongdict[clusternum] += 1
			upos[clusternum] += 1
			upl[clusternum].append((x,y))
		else:
			unrelatedneg += 1
			rightdict[clusternum] += 1
			uneg[clusternum] += 1
			unl[clusternum].append((x,y))

	tmin = 1e8
	tmax = 0
	tpos = 0
	tneg = 0
	for x,y in train:
		d,clusternum = calcEulerDisInClusters(x,y)
		tmin = min(tmin, d)
		tmax = max(tmax, d)

		if d < thresholdrate * finalcosts[clusternum]:
			tpos += 1
		else:
			tneg += 1


	if output:
		for i in xrange(kcluster):
			writeln('Cluster %d'%(i))
			writeln('\trelatedpos %d'%(len(rpl[i])))
			for x,y in rpl[i]:
				writeln('\t\trpos %s %s'%(y.encode('utf-8'),x.encode('utf-8')))

			writeln('\n\trelatedneg %d'%len(rnl[i]))
			for x,y in rnl[i]:
				writeln('\t\trneg %s %s'%(y.encode('utf-8'),x.encode('utf-8')))
			writeln('\n\tunrelatedpos %d'%len(upl[i]))
			for x,y in upl[i]:
				writeln('\t\tupos %s %s'%(y.encode('utf-8'),x.encode('utf-8')))
			writeln('\n\tunrelatedneg %d'%len(unl[i]))
			for x,y in unl[i]:
				writeln('\t\tuneg %s %s'%(y.encode('utf-8'),x.encode('utf-8')))
			writeln('\n')

	writeln('\nThresholdrate: %.2f'%(thresholdrate))
	writeln('Precision: %.6f \tRecall: %.6f \tTrain Precision: %.6f'%( relatedpos*1.0/(relatedpos + unrelatedpos + 1), relatedpos*1.0/(relatedpos + relatedneg + 1), tpos*1.0/(tpos+tneg+1) ))

	pass


def makeGraph(highDdata = [], label = [], col = '', eachlabel = ''):

	pca=PCA(n_components = 2)
	graphdata = pca.fit_transform(highDdata)

	if len(label) != 0:
		graphcluster = [[] for i in range(kcluster)]
		for i in xrange(len(label)):
			graphcluster[ labels_[i] ].append(graphdata[i])
		for i in xrange(len(graphcluster)):
			col = random.choice(colors)
			plt.scatter([x[0] for x in graphcluster[i]], [x[1] for x in graphcluster[i]],label = 'Cluster'+str(i), c=col, s = 20,  edgecolors = 'white')
	else:
		if col == '':
			col = 'black'
		if eachlabel == '':
			eachlabel = 'Test Data'
		plt.scatter([x[0] for x in graphdata], [x[1] for x in graphdata],label = eachlabel, c=col, s = 1,  edgecolors = 'None')
	plt.legend()
	pass



def kmeans_clusters(train):
	global kcluster
	global kms
	global labels_
	global cluster_centers_
	writeln('kmeans, k = %d'%(kcluster))
	labels_ = kms.fit_predict([ model[x[1].encode('utf-8')]-model[x[2].encode('utf-8')] for x in train])
	cluster_centers_ = [x for x in kms.cluster_centers_]

	while True:
		tmpdict = [0 for i in range(kcluster)]
		for i in labels_:
			tmpdict[i] += 1
		writeln('\nTraining Data:\ncluster\tnum')
		for i in xrange(kcluster):
			writeln('%d\t%d'%(i,tmpdict[i]))

		writeln('Need split a cluster? Input the index or anything else to continue:')
		s = raw_input().strip()
		if not s.isdigit():
			break
		k = int(s)
		if k < 0 or k >= kcluster:
			writeln('The index is not right, try again?')
			continue
		tonum = 5
		writeln('Input the number to splite, other thing will be default 5')
		s = raw_input().strip()
		if s.isdigit():
			tonum = int(s)

		tmptrain = []
		for i in xrange(len(labels_)):
			if labels_[i] == k:
				tmptrain.append( model[train[i][1].encode('utf-8')]-model[train[i][2].encode('utf-8')] )
			pass

		tmpkms = cluster.KMeans(tonum)
		tmplabel = tmpkms.fit_predict(tmptrain)
		tmpcenters = tmpkms.cluster_centers_

		indx = 0
		for i in xrange(len(labels_)):
			if labels_[i] == k:
				labels_[i] = kcluster - 1 + tmplabel[indx]
				indx += 1
			elif labels_[i] > k:
				labels_[i] -= 1
		kcluster = kcluster - 1 + tonum

		del cluster_centers_[k]
		for i in tmpcenters:
			cluster_centers_.append(i)
		pass
	return
	# writeln('Need show PCA 2D graph? y to show and anything else to continue')
	# s = raw_input().strip()
	# if s == 'y':
	# 	pca=PCA(n_components = 2)
	# 	graphdata = pca.fit_transform([ model[x[1].encode('utf-8')]-model[x[2].encode('utf-8')] for x in train])
	# 	graphcluster = [[] for i in range(kcluster)]
	# 	for i in xrange(len(train)):
	# 		graphcluster[ labels_[i] ].append(graphdata[i])
	# 	for i in xrange(len(graphcluster)):
	# 		col = random.choice(colors)
	# 		plt.scatter([x[0] for x in graphcluster[i]], [x[1] for x in graphcluster[i]],label = 'Cluster'+str(i), c=col, s = 20,  edgecolors = 'white')
	# 	plt.legend()
	# 	plt.show()
	# return
	pass


def printclusterdetail():
	writeln('cluster    sum rsum usum  rpos rneg upos uneg  Precision Accuracy Recall')
	for i in xrange(kcluster):
		writeln('Cluster%2d:%4d %4d %4d  %4d %4d %4d %4d  %9.4f %8.4f %6.4f'%(
			i,(rpos[i]+rneg[i]+upos[i]+uneg[i]),(rpos[i]+rneg[i]),(upos[i]+uneg[i]),
			rpos[i],rneg[i],upos[i],uneg[i],
			rpos[i]*1.0/(rpos[i] + upos[i] + 1),
			(rpos[i] + uneg[i])*1.0/(rpos[i] + rneg[i] + upos[i] + uneg[i] + 1),
			rpos[i]*1.0/(rpos[i] + rneg[i] + 1)
			)
		)
	writeln('Detail test end')





def deal_test_data(test):
	writeln('Test Data prepair')
	related = [(x[2],x[1]) for x in test]
	unrelated = []
	while len(unrelated) < 3250:
		x,y = random.sample(all_word.items(),2)
		pre = min(len(x[0]),len(y[0]))
		if x[0][:pre] == y[0][:pre]:
			if len(x[1]) >= 2:
				tx,ty = random.sample(x[1],2)
				if tx.encode('utf-8') in model and ty.encode('utf-8') in model:
					unrelated.append((tx,ty))
			elif len(y[1]) >= 2:
				tx,ty = random.sample(y[1],2)
				if tx.encode('utf-8') in model and ty.encode('utf-8') in model:
					unrelated.append((tx,ty))
		else:
			tx = random.choice(x[1])
			ty = random.choice(y[1])
			if tx.encode('utf-8') in model and ty.encode('utf-8') in model:
				unrelated.append((tx,ty))
	writeln('Test Data Kanliou')
	return related,unrelated
	pass

def __main__():
	makeTree()
	root.traverse()
	global alltriple
	for t in triple:
		if len(t[2]) > kcluster*2 or len(t[0])>1 or len(t[1])>1:
			continue
			pass
		for i in t[0]:
			if i.encode('utf-8') not in model:
				continue
			for j in t[1]:
				if j.encode('utf-8') not in model:
					continue
				for k in t[2]:
					if k.encode('utf-8') not in model:
						continue
					if i!=j and j!=k and i!=k:
						alltriple.append((i,j,k))
						# writeln(i.encode('utf-8'),j.encode('utf-8'),k.encode('utf-8')

	writeln('All triple num: %d'%(len(alltriple)))
	if len(alltriple)>3000:
		alltriple = random.sample(alltriple, 3000)

	# sample = random.sample(alltriple, 1391)
	# train = sample[:len(sample)/4]
	# test = sample[len(sample)/4:]


	sample = alltriple
	test = sample[:len(sample)/3]
	train = sample[len(sample)/3:]
	writeln('Train data num: %d\n Test data num: %d'%(len(train),len(test)))


	kmeans_clusters(train)
	if affirm('Show Train Data'):
		makeGraph([ model[x[1].encode('utf-8')]-model[x[2].encode('utf-8')] for x in train], label = labels_)
		plt.show()

	related,unrelated = deal_test_data(test)

	if affirm('Show Train and Test Data'):
		makeGraph([ model[x[1].encode('utf-8')]-model[x[2].encode('utf-8')] for x in train], label = labels_)
		makeGraph([ model[x[1].encode('utf-8')]-model[x[0].encode('utf-8')] for x in related], eachlabel = 'related', col = 'gray')
		makeGraph([ model[x[1].encode('utf-8')]-model[x[0].encode('utf-8')] for x in unrelated], eachlabel = 'unrelated', col = 'black')
		plt.show()


	for i in xrange(kcluster):
		wrongdict[i] = 0
		rightdict[i] = 0
		rpos[i] = 0
		uneg[i] = 0
		upos[i] = 0
		rneg[i] = 0
		sumnum[i] = 0


	group = []
	for i in xrange(kcluster):
		group.append(([],[]))

	for i in xrange(len(train)):
		group[labels_[i]][0].append( model[train[i][2].encode('utf-8')] )
		group[labels_[i]][1].append( model[train[i][1].encode('utf-8')] )

	for i in xrange(kcluster):
		writeln('Cluster %d'%(i))
		phi,cost = calcPhiSGD(group[i][0], group[i][1])
		phis.append( phi )
		finalcosts.append( cost )




	writeln('Detail test start')
	tmpthresholdrate = 1.2
	ForTest(related, unrelated, tmpthresholdrate, [(x[2],x[1]) for x in train])
	writeln('\n\nDict Detail:')
	printclusterdetail()

	writeln('\n\nGenetal test start')
	tmpthresholdrate= 0.1
	while tmpthresholdrate< 3:
		ForTest(related, unrelated, tmpthresholdrate, [(x[2],x[1]) for x in train])
		printclusterdetail()
		tmpthresholdrate += 0.1

	while tmpthresholdrate< 20:
		ForTest(related, unrelated, tmpthresholdrate, [(x[2],x[1]) for x in train])
		printclusterdetail()
		tmpthresholdrate+= 1
	pass

__main__()
