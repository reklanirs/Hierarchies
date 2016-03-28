#!/usr/bin/python2.7
# -*- coding: utf-8 -*- import requests

from dependent import *

class Node(object):
	"""docstring for Node"""
	def __init__(self, _value = [], _keylen = 1, _level = 0):
		self.value = _value
		self.child = {}
		self.keylen = _keylen
		self.level = _level
		


class CilinE(object):
	"""docstring for CilinE"""
	def __init__(self):
		self.root = Node(['ROOT'],1)
		self.all_word = {}
		self.triple = []
		self.alltriple = []
		self.tmptriple = ['','','']

		self.childnum = {}

		self.makeTree()
		self.traverse()

	def makeTree(self):
		#l12
		for i in readfile('CilinE/l12.txt'):
			# writeln(i.encode('utf-8')
			tmp = i.split()
			key = tmp[0]
			value = tmp[1:]
			self.all_word[key] = value
			if len(key) == 1:
				tmp = Node(value,1,1)
				self.root.child[key] = tmp
				pass
			elif len(key) == 2:
				tmp = Node(value,2,2)
				self.root.child[key[0]].child[key[1]] = tmp
				pass
			else:
				writeln('Error '+i)
				pass


		#l3
		for i in readfile('CilinE/l3.txt'):
			tmp = i.split()
			key = tmp[0][:-1]
			value = tmp[1:]
			self.all_word[key] = value
			tmp = Node(value,1,3)
			self.root.child[key[0]].child[key[1]].child[key[2:]] = tmp

		#l45
		for i in readfile('CilinE/l45.txt'):
			tmp = i.split()
			key = tmp[0][:-1]
			symbol = tmp[0][-1]
			value = tmp[1:]
			self.all_word[key] = value

			l4 = key[4]
			l5 = key[5:]
			if l4 not in self.root.child[key[0]].child[key[1]].child[key[2:4]].child:
				tmp = Node([l4],2,4)
				self.root.child[key[0]].child[key[1]].child[key[2:4]].child[l4] = tmp
			# tmp = Node([symbol + i for i in value], 0, 5)
			tmp = Node(value, 0, 5)
			self.root.child[key[0]].child[key[1]].child[key[2:4]].child[l4].child[l5] = tmp

		writeln('MakeTree End')
		pass

	def traverse(self, node = '', t = 0, flag = False):
		if node == '':
			node = self.root

		if node.keylen == 0:
			tmp = len(node.value)
			if tmp not in self.childnum:
				self.childnum[tmp] = 1
			else:
				self.childnum[tmp] += 1
		if node.level == 2 or node.level == 3:
			self.tmptriple[node.level - 2] = node.value
		if node.level == 5:
			self.tmptriple[2] = node.value
			self.triple.append(self.tmptriple[:])

		if flag:
			for i in range(t):
				write('\t')
			for i in node.value:
				write(i.encode('utf-8'))
			writeln('')

		for i,j in node.child.items():
			self.traverse(j, t + 1, flag)
		pass


# c = CilinE()
# c.traverse(flag = True)