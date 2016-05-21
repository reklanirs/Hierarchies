#!/usr/bin/python2.7
# -*- coding: utf-8 -*- 

from graph import *
from dependent import *

class Comparer(object):
    """docstring for Comparer"""
    def __init__(self, _vectors):
        super(Comparer, self).__init__()
        self.vectors = _vectors
        # self.Euclidean_similarity(self.vectors)
        # self.Cosine_similarity(self.vectors)
        print 'Comparer Finished'

    def Euclidean(self, a, b):
        return np.linalg.norm(a-b)

    def Euclidean_similarity(self, vectors):
        values = []
        for i in xrange(len(vectors)-1):
            for j in xrange(i+1, len(vectors)):
                values.append(self.Euclidean(vectors[i],vectors[j]))
        num_diagram(values, 100, xlabel = 'Euclidean dis')
        pass

    def Cosine(self,a,b):
        return np.dot(a,b)/(np.linalg.norm(a) * np.linalg.norm(b))

    def Cosine_similarity(self, vectors):
        values = []
        for i in xrange(len(vectors)-1):
            for j in xrange(i+1, len(vectors)):
                values.append(self.Cosine(vectors[i],vectors[j]))
        num_diagram(values, 100, xlabel = 'Cosine dis')
        pass

    def clustering(self, l):
        p = [ds[0].rp for ds in l]
        eps = []
        for ds in l:
            tmp = []
            for d in ds:
                ep = cconv(d.es,d.eo)
                tmp.append(ep)
            eps.append(tmp)
        
        for i in range(len(l)):
            # values = [ self.Euclidean(p[i], ep)  for ep in eps[i]]
            # values1 = [ self.Euclidean(p[len(l)-i-1], ep)  for ep in eps[i]]
            # dual_num_diagram(values,values1)
            triples = [ (j+0.5, self.Euclidean(p[i], eps[i][j]), self.Euclidean(p[len(l)-i-1], eps[i][j]))  for j in xrange(len(eps[i])) ] 
            dual_bar_diagram(triples, 'indx', '+Y: Number of p1; -Y:Number of p2' )
        pass

    # def ctmp2(self, triples):
    #     tmp = [x[1]-x[2] for x in triples]
    #     num_diagram(tmp,100,'Prob Difference','Number')
    #     # dual_bar_diagram(triples, 'indx', '+Y: Prob of p1; -Y:Prob of p2' )

    # def ctmp3(self, triples):
    #     tmp = [ self.Cosine(x[1],x[2])  for x in triples]
    #     num_diagram(tmp,100,'Cosine_similarity','Number')
