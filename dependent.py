#!/usr/bin/python2.7
# -*- coding: utf-8 -*- 
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
import copy
from random import choice
from bs4 import BeautifulSoup
from bs4 import UnicodeDammit
from lxml import etree
import jieba
import gensim, logging
from sklearn import cluster
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft

colorsbk = ['#FFFAFA','#F8F8FF','#F5F5F5','#DCDCDC','#FFFAF0','#FDF5E6','#FAF0E6','#FAEBD7','#FFEFD5','#FFEBCD','#FFE4C4','#FFDAB9','#FFDEAD','#FFE4B5','#FFF8DC','#FFFFF0','#FFFACD','#FFF5EE','#F0FFF0','#F5FFFA','#F0FFFF','#F0F8FF','#E6E6FA','#FFF0F5','#FFE4E1','#FFFFFF','#000000','#2F4F4F','#696969','#708090','#778899','#BEBEBE','#D3D3D3','#191970','#000080','#6495ED','#483D8B','#6A5ACD','#7B68EE','#8470FF','#0000CD','#4169E1','#0000FF','#1E90FF','#00BFFF','#87CEEB','#87CEFA','#4682B4','#B0C4DE','#ADD8E6','#B0E0E6','#AFEEEE','#00CED1','#48D1CC','#40E0D0','#00FFFF','#E0FFFF','#5F9EA0','#66CDAA','#7FFFD4','#006400','#556B2F','#8FBC8F','#2E8B57','#3CB371','#20B2AA','#98FB98','#00FF7F','#7CFC00','#00FF00','#7FFF00','#00FA9A','#ADFF2F','#32CD32','#9ACD32','#228B22','#6B8E23','#BDB76B','#EEE8AA','#FAFAD2','#FFFFE0','#FFFF00','#FFD700','#EEDD82','#DAA520','#B8860B','#BC8F8F','#CD5C5C','#8B4513','#A0522D','#CD853F','#DEB887','#F5F5DC','#F5DEB3','#F4A460','#D2B48C','#D2691E','#B22222','#A52A2A','#E9967A','#FA8072','#FFA07A','#FFA500','#FF8C00','#FF7F50','#F08080','#FF6347','#FF4500','#FF0000','#FF69B4','#FF1493','#FFC0CB','#FFB6C1','#DB7093','#B03060','#C71585','#D02090','#FF00FF','#EE82EE','#DDA0DD','#DA70D6','#BA55D3','#9932CC','#9400D3','#8A2BE2','#A020F0','#9370DB','#D8BFD8','#FFFAFA','#EEE9E9','#CDC9C9','#8B8989','#FFF5EE','#EEE5DE','#CDC5BF','#8B8682','#FFEFDB','#EEDFCC','#CDC0B0','#8B8378','#FFE4C4','#EED5B7','#CDB79E','#8B7D6B','#FFDAB9','#EECBAD','#CDAF95','#8B7765','#FFDEAD','#EECFA1','#CDB38B','#8B795E','#FFFACD','#EEE9BF','#CDC9A5','#8B8970','#FFF8DC','#EEE8CD','#CDC8B1','#8B8878','#FFFFF0','#EEEEE0','#CDCDC1','#8B8B83','#F0FFF0','#E0EEE0','#C1CDC1','#838B83','#FFF0F5','#EEE0E5','#CDC1C5','#8B8386','#FFE4E1','#EED5D2','#CDB7B5','#8B7D7B','#F0FFFF','#E0EEEE','#C1CDCD','#838B8B','#836FFF','#7A67EE','#6959CD','#473C8B','#4876FF','#436EEE','#3A5FCD','#27408B','#0000FF','#0000EE','#0000CD','#00008B','#1E90FF','#1C86EE','#1874CD','#104E8B','#63B8FF','#5CACEE','#4F94CD','#36648B','#00BFFF','#00B2EE','#009ACD','#00688B','#87CEFF','#7EC0EE','#6CA6CD','#4A708B','#B0E2FF','#A4D3EE','#8DB6CD','#607B8B','#C6E2FF','#B9D3EE','#9FB6CD','#6C7B8B','#CAE1FF','#BCD2EE','#A2B5CD','#6E7B8B','#BFEFFF','#B2DFEE','#9AC0CD','#68838B','#E0FFFF','#D1EEEE','#B4CDCD','#7A8B8B']
colors = ['#E6E6FA', '#00BFFF', '#98FB98', '#8B4513', '#FF1493', '#FFF5EE', '#CDC9A5', '#F0FFFF', '#4F94CD']

model = gensim.models.Word2Vec.load('model/m100.w2v')
fout = open('out.txt','w')
sysstdout = sys.stdout

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

def readfile(_file):
	for i in open(_file):
		yield i.strip().decode('utf-8').strip()

def dis2(x,y):
	return dis(x,y)**2
	pass

def dis(x,y):
	return np.linalg.norm(x-y)
	pass

def makeGraph(l, highDdata = [], label = [], col = '', eachlabel = ''):

	pca=PCA(n_components = 2)
	graphdata = pca.fit_transform(highDdata)

	if len(label) != 0:
		graphcluster = [[] for i in range(l.kcluster)]
		for i in xrange(len(label)):
			graphcluster[ l.labels_[i] ].append(graphdata[i])
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



def ccorr(a, b):
	"""
	Circular correlation of vectors
	Computes the circular correlation of two vectors a and b via their
	fast fourier transforms
	a \ast b = \mathcal{F}^{-1}(\overline{\mathcal{F}(a)} \odot \mathcal{F}(b))
	Parameter
	---------
	a: real valued array (shape N)
	b: real valued array (shape N)
	Returns
	-------
	c: real valued array (shape N), representing the circular
	   correlation of a and b
	"""
	return ifft(np.conj(fft(a)) * fft(b)).real

	
def cconv(a, b):
	"""
	Circular convolution of vectors
	Computes the circular convolution of two vectors a and b via their
	fast fourier transforms
	a \ast b = \mathcal{F}^{-1}(\mathcal{F}(a) \odot \mathcal{F}(b))
	Parameter
	---------
	a: real valued array (shape N)
	b: real valued array (shape N)
	Returns
	-------
	c: real valued array (shape N), representing the circular
	   convolution of a and b
	"""
	return ifft(fft(a) * fft(b)).real




def Euclidean(a, b):
    return np.linalg.norm(a-b)


def Cosine(a,b):
    return np.dot(a,b)/(np.linalg.norm(a) * np.linalg.norm(b))

class Data(object):
	"""( ( (s,es),(o,eo),(p,rp) ), y ) 的list. 其中s|o为任意汉词, p为自定义?关系名, 格式为utf-8 .
	从cilinE解析出来为unicode, 需要encode为utf-8 后才可作为model的key 
	es|eo为 np.array(model.layer1_size), ep为自定义生成的同纬度np.array, y 为标签"""
	def __init__(self, _s = '', _es = [], _o = '', _eo = [], _p = '', _rp = [], _y = 1):
		self.s = _s
		self.es = _es
		self.o = _o
		self.eo = _eo
		self.p = _p
		self.rp = _rp
		self.y = _y
		self.ans = 0
		self.stop,self.otop = [],[]
		self.srank,self.orank = 0,0