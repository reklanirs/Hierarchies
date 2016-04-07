#!/usr/bin/python2.7
# -*- coding: utf-8 -*- import requests
from dependent import *



class Linear(object):
	"""docstring for Linear"""
	def __init__(self):
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


	def cost(self,phi,x,y):
		return dis2(np.dot(x,phi), y)
		pass

	def costAll(self,phi,X,Y):
		return dis2(np.dot(X,phi), Y)/len(X)
		pass


	def calcEulerDisInClusters(self,x,y):
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


	def calcPhiSGD(self,X,Y):
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

	def ForTest(self,related, unrelated, thresholdrate = 1.2, train = [], output = False):
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


	def kmeans_clusters(self,train):
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


	def printclusterdetail(self):
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



