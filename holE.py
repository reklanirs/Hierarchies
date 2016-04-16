#!/usr/bin/python2.7
# -*- coding: utf-8 -*- import requests

from dependent import *
import numpy as np
from numpy.fft import fft, ifft


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


class HolE(object):
	"""docstring for HolE"""
	def __init__(self, _trainData = [], _testData = []):
		self.trainData = _trainData
		self.testData = _testData
		self.w2v = {}
		self.elist,self.plist = set(),set()
		self.extractData(self.trainData)
		self.training(self.trainData)
		writeln('HolE Training End')

		# for p in self.plist:
		# 	print('p:%s\nrp:')
		# 	print self.w2v[p]

		# for d in self.trainData:
		# 	writeln('s:%s o:%s P:%f'%(d.s,d.o,self.calcP(d.es,d.eo,d.rp)))
		# self.testing(self.testData)

		# self.testTopK(self.testData)
		self.testing(self.testData)
		writeln('HolE Testing End')

	def extractData(self, D):
		for d in self.trainData:
			for i,j in ((d.s,d.es),(d.o,d.eo),(d.p,d.rp)):
				self.w2v[i] = j
			self.elist.add(d.s)
			self.elist.add(d.o)
			self.plist.add(d.p)
		pass

	"""存疑: a-b b-c, 同时含有b, 如何更新b? 暂时使用map方法"""
	def training(self, D):
		steps = 500
		for step in xrange(steps):
			for i in xrange(len(D)):
				curData = D[i]
				if curData.y == 1:
					D[i] = self.minimCostFunction1(curData)
			if step%50 == 0 or step == steps - 1:
				writeln('Current Costfunction Value:%f'%self.costFunction1Value())
		pass
	def minimCostFunction1(self, d, u = 0.1):
		d.es,d.eo,d.rp = self.w2v[d.s],self.w2v[d.o],self.w2v[d.p]
		eta = np.dot(d.rp, ccorr(d.es, d.eo))
		if eta>500 or eta<-500:
			print 'eta:',eta
			print '\nrp:',d.rp
			print '\nes:',d.es
			print '\neo:',d.eo
		denominator = 1 + math.exp(eta)
		# if eta > 100:
		# 	return d
		# elif eta < -100:
		# 	denominator = 1
		# else:
		# 	denominator = 1 + math.exp(eta)
		es = d.es + u * ccorr(d.rp, d.eo) / denominator
		eo = d.eo + u * cconv(d.rp, d.es) / denominator
		rp = d.rp + u * ccorr(d.es, d.eo) / denominator
		d.es,d.eo,d.rp = es,eo,rp
		self.w2v[d.s],self.w2v[d.o],self.w2v[d.p] = es,eo,rp
		return d

	def costFunction1Value(self):
		ret = 0.0
		for d in self.trainData:
			ret += math.log(1.0 + math.exp( - np.dot(d.rp, ccorr(d.es, d.eo)) ))
		return ret

	def calcP(self, es,eo,rp):
		return 1.0/(1.0 + math.exp( -(np.dot(rp, ccorr(es, eo))) ) )

	def testing(self, D):
		for d in D:
			d.ans = self.calcP(self.w2v[d.s], self.w2v[d.o], self.w2v[d.p])
			writeln('s:%s o:%s Pro:%f\n'%(d.s,d.o,d.ans))
		pass

	def testTopK(self, D):
		for d in D:
			# s,p  <-  o
			lo = []
			es = self.w2v[d.s]
			rp = self.w2v[d.p]
			m = ccorr(es, rp)
			for o in self.elist:
				eo = self.w2v[o]
				prob = np.dot(eo, m)
				lo.append((o,prob))
			lo.sort(key = lambda x:x[1], cmp = lambda x,y: cmp(float(x),float(y)), reverse = True)
			d.orank = lo.index( (d.o, np.dot(self.w2v[d.o],m)) )
			d.otop = lo[:10]


			# o,p  <-  s
			ls = []
			eo,rp = self.w2v[d.o],self.w2v[d.p]
			m = ccorr(eo, rp)
			for s in self.elist:
				es = self.w2v[s]
				prob = np.dot(es, m)
				ls.append((s,prob))
			ls.sort(key = lambda x:x[1], cmp = lambda x,y: cmp(float(x),float(y)), reverse = True)
			d.srank = ls.index((d.s, np.dot(self.w2v[d.s],m) ))
			d.stop = ls[:10]
		pass

	
	def rank(self,k,l = [1,3,5,10,100,1000,1000000]):
		for i in xrange(len(l)):
			if k<l[i]:
				return i
		return len(l) - 1

	def output(self, r = 0.99):
		l = [0.5,0.6,0.7,0.8,0.85,0.9,0.95,0.99,1.0]
		num = [0 for x in xrange(len(l))]

		A,B,C,D = 0,0,0,0
		Al,Bl,Cl,Dl = [],[],[],[]
		for d in self.testData:
			if d.y == 1:
				num[self.rank(d.ans,l)] += 1
				if d.ans > r:
					A += 1
					Al.append(d)
				else:
					C += 1
					Cl.append(d)
			elif d.y == 0:
				if d.ans >= r:
					B += 1
					Bl.append(d)
				else :
					D += 1
					Dl.append(d)
		writeln('\nr:%f'%r)
		writeln(' sum rsum usum     A    C    B    D  Accuracy  Precision Recall')
		writeln('%4d %4d %4d  %4d %4d %4d %4d %8.4f   %9.4f %6.4f'%(
			(A+C+B+D),(A+C),(B+D),
			A,C,B,D,
			(A + D)*1.0/(A + C + B + D + 1),
			A*1.0/(A + B + 1),
			A*1.0/(A + C + 1)
			)
		)

		for s in 'Al','Bl','Cl','Dl':
			writeln('\n%s Sample:'%s)
			for d in random.sample(eval(s),min(30,len(eval(s)))):
				writeln('\ts:%s o:%s'%(d.s,d.o))

		writeln('Probility:')
		s = sum(num)
		prob = [x for x in num]
		for i in xrange(1,len(l)):
			for j in xrange(i-1,-1,-1):
				prob[j] += prob[i]
		for i in xrange(len(l)):
			prob[i] = prob[i]*1.0/s
		write('   ')
		for i in l:
			write('%5.2f'%i)
		writeln('')
		for i in num:
			write('%5d'%i)
		writeln('\n')
		for i in l[:-1]:
			write(' >%4.2f'%i)
		writeln('')
		for i in prob[:-1]:
			write('%6.3f'%i)
		# writeln('Detail test end')
		pass

	def outputTopK(self):
		#top 1, 3, 5, 10, 100, 1000, other
		l = [1,3,5,10,100,1000,1000000]
		poss = [0 for x in range(len(l))]
		poso = [0 for x in range(len(l))]
		pos = [0 for x in range(len(l))]
		negs = [0 for x in range(len(l))]
		nego = [0 for x in range(len(l))]
		neg = [0 for x in range(len(l))]
		for d in self.testData:
			if d.y == 1:
				poss[self.rank(d.srank)] += 1
				poso[self.rank(d.orank)] += 1
			elif d.y == 0:
				negs[self.rank(d.srank)] += 1
				nego[self.rank(d.orank)] += 1
		posSum = sum(poss)
		negSum = sum(negs)
		poss = [x*1.0/posSum for x in poss]
		poso = [x*1.0/posSum for x in poso]
		negs = [x*1.0/(negSum+1) for x in negs]
		nego = [x*1.0/(negSum+1) for x in nego]

		for i in xrange(1,len(l)):
			poss[i] += poss[i-1]
			poso[i] += poso[i-1]
			negs[i] += negs[i-1]
			nego[i] += nego[i-1]

		for i in xrange(len(l)):
			pos[i] = (poss[i] + poso[i])/2.0
			neg[i] = (negs[i] + nego[i])/2.0

		wrongl = []
		rightl = []
		for d in self.testData:
			if d.orank < 10:
				rightl.append(d)
			elif len(wrongl) < 100:
				wrongl.append(d)

		writeln('\nWring cases:')
		k = 3
		for d in wrongl:
			write('s:%s p:%s o:? want:%s, top%d:'%(d.s, d.p, d.o, k))
			for i in xrange(k):
				write(' ' + d.otop[i][0])
			write('\n')

		writeln('\nRight cases:')
		k = 5
		for d in rightl:
			write('s:%s p:%s o:? want:%s, top%d:'%(d.s, d.p, d.o, k))
			for i in xrange(k):
				write(' ' + d.otop[i][0])
			write('\n')


		write('\nHits at\n    ')
		for i in l:
			write('%8d'%i)

		for l in 'pos','poss','poso':#,'neg','negs','nego':
			write('\n%4s'%l)
			for i in eval(l):
				write('%8.5f'%i)
		writeln('')
		pass

