#!/usr/bin/python2.7
# -*- coding: utf-8 -*- import requests
import os
import re
import sys
import math
import requests
import time
import json
import lxml
import demjson
from random import choice
from bs4 import BeautifulSoup
from bs4 import UnicodeDammit
from lxml import etree
reload(sys)

log = open('log.txt','a')
fsus = open('sus.txt','a+')
ffai = open('fail.txt','a+')
sen = open('sen.txt','a+')
fsus.seek(0,0)
ffai.seek(0,0)

#sys.stdin=open('in.txt','r')
#sys.stdout=open('out.txt','w')

url = 'http://baike.baidu.com/search/word?word='

duoyici = '这是一个<a href="/view/10812277.htm" target="_blank">多义词</a>'

keyword = '久仰'
wordset = set()
susset = set()
faiset = set()
allsenset = set()

def writeln(_s):
	if isinstance(_s, unicode):
		s = _s.encode('utf-8')
	else:
		s = _s
	print s
	log.write(str(s) + '\n')
	log.flush()
	pass

def write(_s):
	if isinstance(_s, unicode):
		s = _s.encode('utf-8')
	else:
		s = _s
	print s,
	log.write(str(s))
	log.flush()
	pass

def request_with_keyword(k = keyword, out = False):
	rel = []
	r = requests.get(url + k)
	if r.content.find(duoyici) != -1:
		rc = r.content
		tmp = rc.find("href=\"/subview/")
		while tmp != -1:
			tmp2 = rc[tmp:].find('>')
			tmpurl = 'http://baike.baidu.com/' + rc[tmp+7 : tmp+tmp2-1]
			tmpre = requests.get(tmpurl)
			rel.append(tmpre)
			
			rc = rc[tmp + tmp2 :]
			tmp = rc.find("href=\"/subview/")
		pass
	else:
		rel.append(r)
	
	#write('\t%d '%(len(rel)))
	s = set()
	for respond in rel:
		soup = BeautifulSoup(respond.content, 'lxml')
		div = soup.find_all('div')
		if out:
			fout = open('out.txt','w')
		for d in div:
			#print d.text.encode('utf-8'),'\n' 
			for i in d.text.split('\n'):
				i = i.strip()

				if len(i)>5:
					if re.match('\\[\\d\\]',i[-3:]):
						i = i[:-3]
					elif re.match('\\[\\d\\d\\]',i[-4:]):
						i = i[:-4]
					i = i.strip()
					if i[-1] in '。”？'.decode('utf-8'):
						s.add(i.strip())
						if out:
							fout.write(i.encode('utf-8') + '\n')
							fout.flush()
			if out:
				# fout.write('#####################\n')
				pass
	return s
	pass

def check_is_chinese(ch):
	if u'\u4e00' <= ch <= u'\u9fff':
		return True
	return False


def extract_words1():
	fin = open('哈工大社会计算与信息检索研究中心同义词词林扩展版.txt','r')
	wordout = open('word1.txt','w')
	while True:
		s = fin.readline()
		if len(s)==0:
			break
		s = s.decode('utf-8')
		#print 's:',s.encode('utf-8')
		for i in s.split(' '):
			#print 'i:',i.encode('utf-8')
			if len(i) > 0 and check_is_chinese(i[0]):
				wordout.write(i.strip().encode('utf-8') + '\n')
				wordout.flush()
	pass	

def extract_words2():
	fin = open('词性-词义_合并结果.txt','r')
	wordout = open('word2.txt','w')
	while True:
		s = fin.readline()
		if len(s)==0:
			break
		s = s.decode('utf-8')
		for i in s.split(' '):
			if len(i) == 0 or not check_is_chinese(i[0]):
				continue
			tmp = ''
			for j in i:
				if check_is_chinese(j):
					tmp += j
					continue
				if tmp!='':	
					wordout.write(tmp.strip().encode('utf-8') + '\n')
					wordout.flush()
					break
	pass	

def download_from_word(wordfile = ''):
	if len(wordfile) == 0:
		return
	global wordset
	fin = open(wordfile,'r')
	while True:
		s = fin.readline()
		if len(s) == 0:
			break
		s = s.strip().decode('utf-8')
		wordset.add(s)

	while True:
		j = fsus.readline()
		if len(j)==0:
			break
		j = j.strip().decode('utf-8')
		susset.add(j)
	while True:
		j = ffai.readline()
		if len(j)==0:
			break
		j = j.strip().decode('utf-8')
		faiset.add(j)


	print 'sused:%d failed:%d  left:%d'%(len(susset),len(faiset),len(wordset)-len(susset)-len(faiset))
	num = 0
	for key in wordset:
		num += 1
		print '%6d '%(num),key.encode('utf-8'),' ',
		if key in susset or key in faiset:
			print '\tignore'
			continue
		s = request_with_keyword(key)
		write('%6s\t'%(key))
		write('%3d'%(len(s)))
		if len(s) > 0:
			sen.write('%s %d\n'%(key.encode('utf-8'),len(s)))
			for i in s:
				sen.write(i.encode('utf-8') + '\n')
			sen.flush()
			fsus.write(key.encode('utf-8') + '\n')
			fsus.flush()
		else:
			ffai.write(key.encode('utf-8') + '\n')
			ffai.flush()
		writeln('\t fin')
	pass




#s = request_with_keyword('看上', False)

# for i in s:
# 	sen.write(i.encode('utf-8') + '\n')
download_from_word('complex.txt')

