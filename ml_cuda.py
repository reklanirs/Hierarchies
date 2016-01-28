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
reload(sys)

all_word = {}
childnum = {}
model = gensim.models.Word2Vec.load('model/m100.w2v')
print 'Model loaded'
triple = []
tmptriple = ['','','']
alltriple = []

kcluster = 10
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
	rt = 0.994
	loopnum = 500
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






def ForTest(related, unrelated, thresholdrate = 0.8, train = [], output = False):
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


	global kms
	global labels_
	global cluster_centers_
	writeln('kmeans, k = %d'%(kcluster))
	labels_ = kms.fit_predict([ model[x[1].encode('utf-8')]-model[x[2].encode('utf-8')] for x in train])
	cluster_centers_ = kms.cluster_centers_
	writeln(labels_)
	# writeln(cluster_centers_)
	tmpdict = {}
	for i in xrange(kcluster):
		tmpdict[i] = 0
		wrongdict[i] = 0
		rightdict[i] = 0
		rpos[i] = 0
		uneg[i] = 0
		upos[i] = 0
		rneg[i] = 0
		sumnum[i] = 0
	for i in labels_:
		tmpdict[i] += 1
	writeln('\nTraining Data:\ncluster\tnum')
	for i,j in tmpdict.items():
		writeln('%d\t%d'%(i,j))

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



	writeln('Detail test start')
	tmpthresholdrate = 1.2
	ForTest(related, unrelated, tmpthresholdrate, [(x[2],x[1]) for x in train], True)
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

