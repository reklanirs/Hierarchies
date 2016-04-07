#!/usr/bin/python2.7
# -*- coding: utf-8 -*- import requests

from dependent import *

class Node(object):
	"""docstring for Node"""
	def __init__(self, _value = [], _keylen = 1, _level = 0):
		self.value = _value #键值对的 值的list 例:[中共 中央]
		self.child = {} #{ 当前层(不含上层value的)独占value : Node }
		self.keylen = _keylen #当前level的下一层节点key的长度
		self.level = _level #l1-l5,l2-l3-l5为重点,root为l0



class CilinE(object):
	"""docstring for CilinE"""
	def __init__(self):
		self.root = Node(['ROOT'],1) #CilinE的根节点
		self.all_word = {} #所有有值节点的 {从ROOT到此整个key : 该节点的value的list}
		self.triple = [] # triple存放了全部l2-l3-l5层value的三元组(list)
		self.tmptriple = ['','',''] #临时变量

		self.childnum = {} #通过traverse获得,叶子节点长为l的节点 { l : 长为l的叶子节点数 }

		self.makeTree()
		self.traverse(flag = False)

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

	def traverse(self, node = None, t = 0, flag = False):
		if node == None:
			node = self.root

		if node.keylen == 0:
			tmp = len(node.value)
			if tmp not in self.childnum:
				self.childnum[tmp] = 1
			else:
				self.childnum[tmp] += 1
		# triple存放了l2-l3-l5层的三元组. root为l02
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
# print len(c.triple)
# for i in random.sample(c.triple, 10):
# 	for j in i:
# 		for k in j:
# 			print k.encode('utf-8'),
# 		print ''
# 	print '\n'

# p = '属于'
# print type(p)
# print p
# print p.encode('utf-8')