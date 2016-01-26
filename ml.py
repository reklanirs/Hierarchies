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
reload(sys)
# sys.stdout = open('out.txt','w')

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
truepos = {}
trueneg = {}
falsepos = {}
falseneg = {}

sysstdout = sys.stdout


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
				print '\t',
			for i in self.value:
				print i.encode('utf-8'),
			print ''
		
		for i,j in self.child.items():
			j.traverse(t + 1, flag)
		pass
		
root = node(['ROOT'],1)

def readfile(_file):
	for i in open(_file):
		yield i.strip().decode('utf-8').strip()

def dis2(x,y):
	ret = 0.
	for i in xrange(len(x)):
		ret += (x[i] - y[i])**2
	return ret

def dis(x,y):
	return math.sqrt(dis2(x,y))

def makeTree():
	global root
	#l12
	for i in readfile('CilinE/l12.txt'):
		# print i.encode('utf-8')
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
			print 'Error ',i
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

	print 'MakeTree End'
	pass




def MatrixCorPlus(l1,l2):
	if len(l1) == 0 or len(l2) == 0:
		print 'Matrix is Empty!'
		return 0
	if len(l1)!= len(l2) or len(l1[0])!=len(l2[0]):
		print 'Demension is different!'
		return 0
	ret = 0.0
	for i in xrange(len(l1)):
		for j in xrange(len(l1[0])):
			ret += l1[i][j] * l2[i][j]
	return ret


def cost(phi,x,y):
	D = len(x)
	v = np.zeros(D)
	for i in xrange(D):
		tmp = 0.0
		for j in xrange(D):
			tmp += x[j]*phi[j][i]
		v[i] = tmp
	return dis2(v,y)

def costAll(phi,X,Y):
	ret = 0.
	n = len(X)
	for i in xrange(n):
		ret += cost(phi,X[i],Y[i])
	return ret/n


def calcEulerDisInClusters(x,y):
	vector_x = model[x.encode('utf-8')]
	vector_y = model[y.encode('utf-8')]
	difference = vector_y - vector_x
	mindis = 1e10
	mink = -1
	for i in xrange(kcluster):
		center = cluster_centers_[i]
		tmpdis = dis2(difference, center)
		if tmpdis < mindis:
			mindis = tmpdis
			mink = i

	return cost(phis[mink], vector_x, vector_y),mink


def calcPhiSGD(X,Y):
	D = len(X[0])
	phi = np.random.sample((D,D))
	eps = 1e-3
	r = 0.1
	rt = 0.9965
	loopnum = 300
	for step in xrange(loopnum):
		# if costAll(phi, X, Y) < eps:
		# 	print 'Less than eps'
		# 	break
		if step%50 == 0:
			print '\tStep %d %f'%(step,costAll(phi, X, Y))
		
		tmp = random.choice(xrange(len(X)))
		x = X[tmp]
		y = Y[tmp]
		for u in xrange(D):
			for v in xrange(D):
				jt = 0.0
				for k in xrange(D):
					jt += x[k]*phi[k][v]
				jt = 2.0 * x[u] * (jt - y[v])
				
				phi[u][v] -= r * jt
		
		r *= rt
		pass
	cost = costAll(phi, X, Y)
	print '\tStep %d %f'%(loopnum,cost)
	return phi,cost






def ForTest(related, unrelated, thresholdrate = 0.8, train = []):
	relatedpos = 0
	relatedneg = 0
	unrelatedpos = 0
	unrelatedneg = 0

	rmin = 1e8
	rmax = 0
	for x,y in related:
		d,clusternum = calcEulerDisInClusters(x,y)
		rmin = min(rmin, d)
		rmax = max(rmax, d)
		# print 'r',d

		if d < thresholdrate * finalcosts[clusternum]:
			relatedpos += 1
			rightdict[clusternum] += 1
			truepos[clusternum]+=1
		else:
			relatedneg += 1
			wrongdict[clusternum] += 1
			falseneg[clusternum] += 1

	umin = 1e8
	umax = 0
	for x,y in unrelated:
		d,clusternum = calcEulerDisInClusters(x,y)
		umin = min(umin, d)
		umax = max(umax, d)
		# print 'u',d

		if d < thresholdrate * finalcosts[clusternum]:
			unrelatedpos += 1
			wrongdict[clusternum] += 1
			falsepos[clusternum] += 1
		else:
			unrelatedneg += 1
			rightdict[clusternum] += 1
			trueneg[clusternum] += 1

	tmin = 1e8
	tmax = 0
	tpos = 0
	tneg = 0
	for x,y in train:
		d,clusternum = calcEulerDisInClusters(x,y)
		tmin = min(tmin, d)
		tmax = max(tmax, d)
		# print 'u',d

		if d < thresholdrate * finalcosts[clusternum]:
			tpos += 1
		else:
			tneg += 1

	# print '\nrelated\nmin:%f\nmax:%f\n'%(rmin,rmax)
	# print 'unrelated\nmin:%f\nmax:%f\n'%(umin,umax)

	print '\nThresholdrate: %.2f'%(thresholdrate)
	print 'Precision: %.6f \tRecall: %.6f \tTrain Precision: %.6f'%( relatedpos*1.0/(relatedpos + unrelatedpos + 1), relatedpos*1.0/(relatedpos + relatedneg + 1), tpos*1.0/(tpos+tneg+1) )
	pass


def __init():
	pass


def __main__():
	makeTree()
	root.traverse()
	for t in triple:
		if len(t[2]) > kcluster or len(t[0])>1 or len(t[1])>1:
			continue
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
						# print i.encode('utf-8'),j.encode('utf-8'),k.encode('utf-8')

	# sample = random.sample(alltriple, 1391)
	# train = sample[:len(sample)/4]
	# test = sample[len(sample)/4:]


	sample = alltriple
	test = sample[:len(sample)/3]
	train = sample[len(sample)/3:]



	global kms
	global labels_
	global cluster_centers_
	print 'kmeans, k = %d'%(kcluster)
	labels_ = kms.fit_predict([ model[x[1].encode('utf-8')]-model[x[2].encode('utf-8')] for x in train])
	cluster_centers_ = kms.cluster_centers_
	print labels_
	# print cluster_centers_
	tmpdict = {}
	for i in xrange(kcluster):
		tmpdict[i] = 0
		wrongdict[i] = 0
		rightdict[i] = 0
		truepos[i] = 0
		trueneg[i] = 0
		falsepos[i] = 0
		falseneg[i] = 0
	for i in labels_:
		tmpdict[i] += 1
	print '\nTraining Data:\ncluster\tnum'
	for i,j in tmpdict.items():
		print '%d\t%d'%(i,j)

	group = []
	for i in xrange(kcluster):
		group.append(([],[]))

	for i in xrange(len(train)):
		group[labels_[i]][0].append( model[train[i][2].encode('utf-8')] )
		group[labels_[i]][1].append( model[train[i][1].encode('utf-8')] )

	for i in xrange(kcluster):
		print 'Cluster %d'%(i+1)
		phi,cost = calcPhiSGD(group[i][0], group[i][1])
		phis.append( phi )
		finalcosts.append( cost )



	print 'Test Data prepair'
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
	print 'Test Data Kanliou'

	
	sys.stdout = open('out.txt','w')

		
	tmpthresholdrate = 0.8
	ForTest(related, unrelated, tmpthresholdrate, [(x[2],x[1]) for x in train])
	# sys.stdout = sysstdout
	print '\n\nDict Detail:'
	print 'cluster\ttruepos\ttrueneg\tfalsepos\tfalseneg\tprecision\tcallback'
	for i in xrange(kcluster):
		# print 'Cluster%2d: %d\t%d\t%f\n\n'%(i,rightdict[i],wrongdict[i],(rightdict[i]*1.0/(rightdict[i]+wrongdict[i])) )
		print 'Cluster%2d: %d\t%d\t%d\t%d\t%f\t%f\n\n'%(i,truepos[i],trueneg[i],falsepos[i],falseneg[i],truepos[i]*1.0/(truepos[i]+falsepos[i]+1), truepos[i]*1.0/(truepos[i] + falseneg[i] + 1) )

	tmpthresholdrate= 0.85
	while tmpthresholdrate< 10:
		ForTest(related, unrelated, tmpthresholdrate, [(x[2],x[1]) for x in train])
		tmpthresholdrate+= 0.05

	while tmpthresholdrate< 50:
		ForTest(related, unrelated, tmpthresholdrate, [(x[2],x[1]) for x in train])
		tmpthresholdrate+= 0.5
	pass

__main__()

