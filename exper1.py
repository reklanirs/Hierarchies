#!/usr/bin/python2.7
# -*- coding: utf-8 -*- import requests
from dependent import *
from cilinE import Node,CilinE
from linear import Linear
from vectorCompare import Comparer

c = CilinE()

def deal_test_data(test):
	writeln('Test Data prepair')
	related = [(x[2],x[1]) for x in test]
	unrelated = []
	while len(unrelated) < 3250:
		x,y = random.sample(c.all_word.items(),2)
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
	alltriple = []

	for t in c.triple:
		if len(t[2]) > 10 or len(t[0])>1 or len(t[1])>1:
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


	l = Linear()


	l.kmeans_clusters(train)
	if affirm('Show Train Data'):
		makeGraph(l, [ model[x[1].encode('utf-8')]-model[x[2].encode('utf-8')] for x in train], label = l.labels_)
		plt.show()

	related,unrelated = deal_test_data(test)

	if affirm('Show Train and Test Data'):
		makeGraph(l, [ model[x[1].encode('utf-8')]-model[x[2].encode('utf-8')] for x in train], label = l.labels_)
		makeGraph(l, [ model[x[1].encode('utf-8')]-model[x[0].encode('utf-8')] for x in related], eachlabel = 'related', col = 'gray')
		makeGraph(l, [ model[x[1].encode('utf-8')]-model[x[0].encode('utf-8')] for x in unrelated], eachlabel = 'unrelated', col = 'black')
		plt.show()


	for i in xrange(l.kcluster):
		l.wrongdict[i] = 0
		l.rightdict[i] = 0
		l.rpos[i] = 0
		l.uneg[i] = 0
		l.upos[i] = 0
		l.rneg[i] = 0
		l.sumnum[i] = 0


	group = []
	for i in xrange(l.kcluster):
		group.append(([],[]))

	for i in xrange(len(train)):
		group[l.labels_[i]][0].append( model[train[i][2].encode('utf-8')] )
		group[l.labels_[i]][1].append( model[train[i][1].encode('utf-8')] )

	for i in xrange(l.kcluster):
		writeln('Cluster %d'%(i))
		phi,cost = l.calcPhiSGD(group[i][0], group[i][1])
		l.phis.append( phi )
		l.finalcosts.append( cost )




	writeln('Detail test start')
	tmpthresholdrate = 1.2
	l.ForTest(related, unrelated, tmpthresholdrate, [(x[2],x[1]) for x in train])
	writeln('\n\nDict Detail:')
	l.printclusterdetail()

	writeln('\n\nGenetal test start')
	tmpthresholdrate= 0.1
	while tmpthresholdrate< 3:
		l.ForTest(related, unrelated, tmpthresholdrate, [(x[2],x[1]) for x in train])
		l.printclusterdetail()
		tmpthresholdrate += 0.1

	while tmpthresholdrate< 20:
		l.ForTest(related, unrelated, tmpthresholdrate, [(x[2],x[1]) for x in train])
		l.printclusterdetail()
		tmpthresholdrate+= 1
	pass

__main__()