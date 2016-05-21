#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

from dependent import *
import graph

class HolE(object):
	"""docstring for HolE"""
	def __init__(self, _train_pos = [], _train_neg = [], _test_pos = [], _test_neg = [], _u = 0.01, _lmd = 0.000001):
		self.trainData_pos,self.trainData_neg = _train_pos,_train_neg
		self.testData_pos,self.testData_neg = _test_pos,_test_neg

		self.u = _u
		self.lmd = _lmd

		self.w2v = {}
		self.elist,self.plist = set(),set()
		self.extractData()

		self.training()	
		writeln('HolE Training End')

		self.testing()
		writeln('HolE Testing End')

	def extractData(self, D = ''):
		if D == '':
			D = self.trainData_pos+self.trainData_neg
		for d in D:
			for i,j in ((d.s,d.es),(d.o,d.eo),(d.p,d.rp)):
				self.w2v[i] = j
			self.elist.add(d.s)
			self.elist.add(d.o)
			self.plist.add(d.p)
		pass

	"""存疑: a-b b-c, 同时含有b, 如何更新b? 暂时使用map方法"""
	def training(self):
		steps = 150
		# one = random.choice(self.trainData_pos)
		# print '\nChoice s:%s o:%s'%(one.s,one.o)
		for step in xrange(steps):
			# print 's:%s o:%s Cosine:%f'%(one.s,one.o,Cosine(one.es, one.eo))
			for i in xrange(len(self.trainData_pos)):
				curData = self.trainData_pos[i]
				self.trainData_pos[i] = self.minimCostFunction1(curData)

			tmp = [(d,self.sigma(d)) for d in self.trainData_pos]
			tmp.sort(key = lambda x:x[1])

			for i in xrange(len(self.trainData_neg)):
				self.trainData_neg[i] = self.minimCostFunction2(tmp[0][0],self.trainData_neg[i])

			if step%20 == 0 or step == steps - 1:
				writeln('Current Costfunction Value:%f'%self.costFunction1Value())

			self.u*=0.98
			self.lmd *= 0.98
		pass


	def minimCostFunction1(self, d):
		d.es,d.eo,d.rp = self.w2v[d.s],self.w2v[d.o],self.w2v[d.p]
		c = ccorr(d.es, d.eo)
		eta = np.dot(d.rp, c)
		try:
			denominator = 1 + math.exp(eta)
			pass
		except Exception, e:
			print '\neta:',eta
			print '\nlmd:',self.lmd
			print '\nccorr:',c
			print '\nd.rp:',d.rp
			raise e

		es = d.es + self.u * ccorr(d.rp, d.eo) / denominator - self.lmd*d.es
		eo = d.eo + self.u * cconv(d.rp, d.es) / denominator - self.lmd*d.eo
		rp = d.rp + self.u * ccorr(d.es, d.eo) / denominator - self.lmd*d.rp

		d.es,d.eo,d.rp = es,eo,rp
		self.w2v[d.s],self.w2v[d.o],self.w2v[d.p] = es,eo,rp
		return d

	def sigma(self, d):
		return 1.0/(1.0 + math.exp(-1.0*(np.dot(d.rp,ccorr(d.es, d.eo)))))
	def minimCostFunction2(self, dplus, dminus, gama = 0.1):
		if gama + self.sigma(dminus) <= self.sigma(dplus):
			return dminus
		d = dminus
		eta = np.dot(d.rp,ccorr(d.es, d.eo))
		es = d.es - self.lmd * ccorr(d.rp, d.eo) / ( 2.0 + math.exp(eta) + math.exp(-eta))
		eo = d.eo - self.lmd * cconv(d.rp, d.es) / ( 2.0 + math.exp(eta) + math.exp(-eta))
		rp = d.rp - self.lmd * ccorr(d.es, d.eo) / ( 2.0 + math.exp(eta) + math.exp(-eta))
		d.es,d.eo,d.rp = es,eo,rp
		return d

	def costFunction1Value(self):
		ret = 0.0
		for d in self.trainData_pos:
			ret += math.log(1.0 + math.exp( - np.dot(d.rp, ccorr(d.es, d.eo)) ))
		return ret

	def calcP(self, es,eo,rp):
		return 1.0/(1.0 + math.exp( -(np.dot(rp, ccorr(es, eo))) ) )

	def testing(self, D = ''):
		if D == '':
			D = self.testData_pos + self.testData_neg
		for d in D:
			d.ans = self.calcP(self.w2v[d.s], self.w2v[d.o], self.w2v[d.p])
			# writeln('s:%s o:%s Pro:%f\n'%(d.s,d.o,d.ans))
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

	def output(self, r = 0.99, prob = False):
		l = [0.5,0.6,0.7,0.8,0.85,0.9,0.95,0.96,0.97,0.98,0.99,1.0]
		num = [0 for x in xrange(len(l))]

		A,B,C,D = 0,0,0,0
		Al,Bl,Cl,Dl = [],[],[],[]
		for d in self.testData_pos:
			num[self.rank(d.ans,l)] += 1
			if d.ans > r:
				A += 1
				Al.append(d)
			else:
				C += 1
				Cl.append(d)
		for d in self.testData_neg:			
			if d.ans >= r:
				B += 1
				Bl.append(d)
			else :
				D += 1
				Dl.append(d)
		writeln('r:%f Accuracy:%.4f  Precision:%.3f  Recall:%.3f'%(r,(A + D)*1.0/(A + C + B + D + 1),
			A*1.0/(A + B + 1),
			A*1.0/(A + C + 1)))
		# writeln('\nr:%f'%r)
		# writeln(' sum rsum usum     A    C    B    D  Accuracy  Precision Recall')
		# writeln('%4d %4d %4d  %4d %4d %4d %4d %8.4f   %9.4f %6.4f'%(
		# 	(A+C+B+D),(A+C),(B+D),
		# 	A,C,B,D,
		# 	(A + D)*1.0/(A + C + B + D + 1),
		# 	A*1.0/(A + B + 1),
		# 	A*1.0/(A + C + 1)
		# 	)
		# )

		# for s in 'Al','Bl','Cl','Dl':
		# 	writeln('\n%s Sample:'%s)
		# 	for d in random.sample(eval(s),min(30,len(eval(s)))):
		# 		writeln('\ts:%s o:%s'%(d.s,d.o))

		if prob:
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

	def makeClusterData(self):
		tmp = {}
		for d in self.trainData:
			if d.y == 0:
				continue
			if d.p not in tmp:
				tmp[d.p] = set()
			tmp[d.p].add(d)
		l = []
		for p,s in tmp.items():
			l.append(random.sample(s,100))
		return l

	# def tmp2(self):
	# 	triples = []
	# 	indx = 0
	# 	for d in self.trainData:
	# 		print 's:%s e:%s  '%(d.s, d.o),
	# 		tmp = [indx]
	# 		for p in self.plist:
	# 			prob = self.calcP(d.es,d.eo,self.w2v[p])
	# 			tmp.append(prob)
	# 			print 'p%s:%f  '%(p, prob ),
	# 		print ' \t\tDifference: %f'%(tmp[1]-tmp[2])
	# 		triples.append(tmp)
	# 	return triples	
	# 	pass

	# def tmp3(self):
	# 	triples = []
	# 	indx = 0
	# 	for d in self.trainData:
	# 		tmp = [indx]
	# 		indx+=1
	# 		tmp.append(d.es)
	# 		tmp.append(d.eo)
	# 		triples.append(tmp)
	# 	return triples	
	# 	pass
