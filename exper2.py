#!/usr/bin/python2.7
# -*- coding: utf-8 -*- 

# subclass disjoint domain range

from dependent import *
from cilinE import Node,CilinE
from linear import Linear
from holE import HolE
from vectorCompare import Comparer




def deal_test_data(test):
	writeln('Test Data prepair')
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
	writeln('Test Data Kanliou')
	return related,unrelated
	pass


"""( ( (s,es),(o,eo),(p,rp) ), y ) 的list. 其中s|o为任意汉词, p为自定义?关系名, 格式为utf-8 .
从cilinE解析出来为unicode, 需要encode为utf-8 后才可作为model的key 
es|eo为 np.array(model.layer1_size), ep为自定义生成的同纬度np.array, y 为标签"""
def makeData():
	# 2 - 3 - 5
	c = CilinE()
	train_pos,train_neg,test_pos,test_neg = [],[],[],[]
	p,p1 = '包含','无关'
	rp,rp1 = model[p],model[p1]
	l35map = {}
	for tri in c.triple:
		t = [x[0].encode('utf-8') for x in tri]
		if not (t[0] in model and t[1] in model and t[2] in model):
			continue
		
		if not t[1] in l35map:
			l35map[t[1]] = set()
		l35map[t[1]].add(t[2])

		A,B,C = t[0],t[1],t[2]
		ma,mb,mc = model[A],model[B],model[C]
		dab,dac = Data(A,ma, B,mb, p,rp, 1), Data(A,ma, C,mc, p,rp, 1)
		dba,dbc = Data(B,mb, A,ma, p,rp, 0), Data(B,mb, C,mc, p,rp, 1)
		dca,dcb = Data(C,mc, A,ma, p,rp, 0), Data(C,mc, B,mb, p,rp, 0)

		train_pos += [dab, dbc]
		train_neg += [dba, dcb] # dca ?
		test_pos += [dac]
		test_neg += [dca]

		if len(train_pos) > 5000:
			break
	
	l3vector = [i for i,j in l35map.items()]
	while len(test_neg) < len(train_neg):
		i = random.randint(0, len(l3vector) - 1)
		j = random.randint(0, len(l3vector) - 1)
		try:
			si = l3vector[i]
			sj = l3vector[j]
		except Exception, e:
			continue
			raise e
		d = Data(si,model[si],sj,model[sj],p,rp,0)
		test_neg.append(d)

	# l3vector = [i for i,j in l35map.items()]
	# allunrelated = []
	# for i in xrange(len(l3vector) - 1):
	# 	for j in xrange(i+1, len(l3vector)):
	# 		e1,e2 = l3vector[i],l3vector[j]
	# 		d1 = Data(e1,model[e1],e2,model[e2],p1,rp1,1)
	# 		d2 = Data(e2,model[e2],e1,model[e1],p1,rp1,1)
	# 		allunrelated += [d1,d2]
	# print 'Before add unrelated, len(train)=%d, len(test)=%d'%(len(train),len(test))
	# train += random.sample(allunrelated, min(len(allunrelated),max(len(test)-len(train),1000)))

	print 'Data finish. %d+%d train cases, %d+%d test cases'%(len(train_pos),len(train_neg),len(test_pos),len(test_neg) )
	return train_pos,train_neg,test_pos,test_neg
	pass

def makeData2():
	# 0 - 1,2 - 3
	c = CilinE()
	train,test = [],[]
	p,p1 = '包含','无关'
	rp,rp1 = model[p],model[p1]
	l35map = {}
	filtered = filter(lambda x:len(x[1])>1, c.triple)
	for tri in random.sample(filtered, min(5000,len(filtered))):
		t = [tri[0][0].encode('utf-8'),tri[1][0].encode('utf-8'),tri[1][1].encode('utf-8'),tri[2][0].encode('utf-8')]
		if not (t[0] in model and t[1] in model and t[2] in model and t[3] in model):
			continue
		d1 = Data(t[0],model[t[0]],t[1],model[t[1]],p,rp,1)
		d2 = Data(t[2],model[t[2]],t[3],model[t[3]],p,rp,1)
		d3 = Data(t[0],model[t[0]],t[2],model[t[2]],p,rp,1)
		train.append(d1)
		train.append(d2)
		test.append(d3)

		if not t[1] in l35map:
			l35map[t[1]] = set()
		l35map[t[1]].add(t[2])
	
	while len(test) < len(train):
		i = random.randint(0, len(train) - 1)
		j = random.randint(0, len(train) - 1)
		try:
			si = train[i].s
			sj = train[j].s
		except Exception, e:
			continue
			raise e
		d = Data(si,model[si],sj,model[sj],p,rp,0)
		test.append(d)

	l3vector = [i for i,j in l35map.items()]
	for i in xrange(len(l3vector) - 1):
		for j in xrange(i+1, len(l3vector)):
			e1,e2 = l3vector[i],l3vector[j]
			d1 = Data(e1,model[e1],e2,model[e2],p1,rp1,1)
			d2 = Data(e2,model[e2],e1,model[e1],p1,rp1,1)
			train.append(d1)
			train.append(d2)

	print 'Data finish. %d train cases, %d test cases'%(len(train),len(test))
	return train,test
	pass



if __name__ == '__main__':


	#为了便于扩展, train,test应该是 [Data] 类型
	train_pos,train_neg,test_pos,test_neg =  makeData()

	u = 0.0001	
	while u<1:
		lmd = 0.000001
		while lmd<0.0011:
			writeln('')
			for i in xrange(64):
				write('#')
			writeln('\n\nu:%.10f lmd:%.10f'%(u,lmd))
			holE = HolE(train_pos,train_neg,test_pos,test_neg,u,lmd)

			r = 0.5
			while r<0.999:
				holE.output(r)
				r += 0.01
			holE.output(0.95, True)

			lmd *= 10
		u *= 10
		pass

	# 修改p的迭代方式
	# 使用topk的测试方式
	# HolE 部分
	# holE = HolE(train_pos,train_neg,test_pos,test_neg)
	

	r = 0.5
	while r<0.99:
		holE.output(r)
		r += 0.01
	holE.output(0.95, True)

