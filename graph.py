#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

from pylab import *

def bar_diagram(_pairs, _n = 10, _xlabel='', _ylabel=''):
    pairs = _pairs[:]
    pairs.sort(key = lambda x : x[0])
    X = [x[0] for x in pairs]
    Y = [x[1] for x in pairs]
    bar(X, Y, width = X[1]-X[0], facecolor='#9999ff', edgecolor='white')
    ylim(min(Y), max(Y)*1.2)
    plt.xlabel(_xlabel)
    plt.ylabel(_ylabel)
    show()

def num_diagram(values, n = 10, xlabel = 'Distance', ylabel = 'Number'):
    step = (max(values) - min(values))*1.0/n
    nums = [0 for x in range(n+1)]
    lowest = min(values)
    for i in values:
        nums[ int((i - lowest)/step) ] += 1
    pairs = []
    for i in range(n):
        pairs.append((lowest + step*0.5 + i*step, nums[i]))
    bar_diagram(_pairs = pairs, _n = n, _xlabel = xlabel, _ylabel = ylabel)


def dual_bar_diagram(_triples, _xlabel='', _ylabel=''):
    triples = _triples[:]
    triples.sort(key = lambda x : x[0])
    X,Y,Y1 = [x[0] for x in triples],[x[1] for x in triples],[-x[2] for x in triples]
    bar(X, Y, width = X[1]-X[0], facecolor='#9999ff', edgecolor='white')
    bar(X, Y1, width = X[1]-X[0], facecolor='#ff9999', edgecolor='white')
    Y = max(max(Y),max(Y1))
    ylim(-Y*1.2, Y*1.2)
    plt.xlabel(_xlabel)
    plt.ylabel(_ylabel)
    show()


def dual_num_diagram(values, values1, n = 20, xlabel = 'Distance', ylabel = '+Y: Number of p1; -Y:Number of p2'):
    biggest,lowest = max(max(values),max(values1)),min(min(values),min(values1))
    step = (biggest - lowest )*1.0/n
    nums,nums1 = [0 for x in range(n+1)],[0 for x in range(n+1)]
    for i in values:
        nums[ int((i - lowest)/step) ] += 1
    for i in values1:
        nums1[ int((i - lowest)/step) ] += 1
    triples = []
    for i in range(n):
        triples.append((lowest + step*0.5 + i*step, nums[i], nums1[i]))
    dual_bar_diagram(_triples = triples,_xlabel = xlabel, _ylabel = ylabel)
