#!/usr/bin/python2.7
# -*- coding: utf-8 -*- import requests

from dependent import *
from cilinE import Node,CilinE
from linear import Linear
from holE import HolE




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
	c = CilinE()
	train,test = [],[]
	# p1,p2,p3 = '属于1','属于2','属于3'
	# rp1,rp2,rp3 = np.random.sample(model.layer1_size)/100.0,np.random.sample(model.layer1_size)/100.0,np.random.sample(model.layer1_size)/100.0
	# i = -10;
	# while i < len(c.triple):
	# 	tri = c.triple[i]
	# 	t = [x[0].encode('utf-8') for x in tri]
	# 	if (not (t[0] in model and t[1] in model and t[2] in model)) or (t[0] == t[1] or t[0] == t[2] or t[1] == t[2]) :
	# 		i+=10
	# 		continue

	# 	t0 = model[t[0]]/100.0
	# 	t1 = model[t[1]]/100.0
	# 	t2 = model[t[2]]/100.0
	# 	# t0 = np.random.sample(model.layer1_size)/10.0
	# 	# t1 = np.random.sample(model.layer1_size)/10.0
	# 	# t2 = np.random.sample(model.layer1_size)/10.0
	# 	d1 = Data(t[2],t2,t[1],t1,p1,rp1,1)
	# 	d2 = Data(t[1],t1,t[0],t0,p2,rp2,1)
	# 	d3 = Data(t[2],t2,t[0],t0,p3,rp3,1)
	# 	# d1 = Data(t[2],np.random.sample(model.layer1_size)/10.0,t[1],np.random.sample(model.layer1_size)/10.0,p1,rp1,1)
	# 	# d2 = Data(t[1],np.random.sample(model.layer1_size)/10.0,t[0],np.random.sample(model.layer1_size)/10.0,p2,rp2,1)
	# 	# d3 = Data(t[2],np.random.sample(model.layer1_size)/10.0,t[0],np.random.sample(model.layer1_size)/10.0,p3,rp3,1)
	# 	# if i%3 == 0:
	# 	# 	train.append(d1)
	# 	# 	train.append(d2)
	# 	# 	test.append(d3)
	# 	# elif i%3 == 1:
	# 	# 	train.append(d1)
	# 	# 	train.append(d3)
	# 	# 	test.append(d2)
	# 	# elif i%3 == 2:
	# 	# 	train.append(d3)
	# 	# 	train.append(d2)
	# 	# 	test.append(d1)
	# 	train.append(d1)
	# 	train.append(d2)
	# 	test.append(d3)
	# 	if len(train) > 6000:
	# 		break
	# 	i += random.randint(5,10)



	p = '包含'
	rp = model[p]
	for tri in c.triple:
		t = [x[0].encode('utf-8') for x in tri]
		if not (t[0] in model and t[1] in model and t[2] in model):
			continue
		d1 = Data(t[0],model[t[0]],t[1],model[t[1]],p,rp,1)
		d2 = Data(t[1],model[t[1]],t[2],model[t[2]],p,rp,1)
		d3 = Data(t[0],model[t[0]],t[2],model[t[2]],p,rp,1)
		train.append(d1)
		train.append(d2)
		test.append(d3)
		if len(train) > 5000:
			break
	
	while len(test) < 5000:
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

	print 'Data finish. %d train cases, %d test cases'%(len(train),len(test))
	return train,test
	pass

def makeData2():
	c = CilinE()
	train,test = [],[]
	p = '包含'
	rp = model[p]
	filtered = filter(lambda x:len(x[1])>1, c.triple)
	for tri in random.sample(filtered, min(5000,len(filtered))):
		# 0 - 1,2 - 3
		t = [tri[0][0].encode('utf-8'),tri[1][0].encode('utf-8'),tri[1][1].encode('utf-8'),tri[2][0].encode('utf-8')]
		if not (t[0] in model and t[1] in model and t[2] in model and t[3] in model):
			continue
		d1 = Data(t[0],model[t[0]],t[1],model[t[1]],p,rp,1)
		d2 = Data(t[2],model[t[2]],t[3],model[t[3]],p,rp,1)
		d3 = Data(t[0],model[t[0]],t[2],model[t[2]],p,rp,1)
		train.append(d1)
		train.append(d2)
		test.append(d3)
	
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

	print 'Data finish. %d train cases, %d test cases'%(len(train),len(test))
	return train,test
	pass



if __name__ == '__main__':
	#为了便于扩展, train,test应该是 [Data] 类型
	train,test =  makeData2()

	# 修改p的迭代方式
	# 使用topk的测试方式
	# HolE 部分
	holE = HolE(train,test)
	holE.output()

	# r = 0.4
	# while r < 1.0:
	# 	holE.output(r)
	# 	r += 0.01

	# Linear 部分
	# linear = Linear(train,test)
	# linear.output()
