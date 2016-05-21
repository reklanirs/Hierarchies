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

