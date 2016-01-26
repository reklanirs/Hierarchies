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

def read_sentences(file):
    fin = open(file,'r')
    while True:
        line = fin.readline()
        if len(line) == 0:
            break
        if len(line.strip().decode('utf-8').split()) != 2:
        	print line
        key,num = line.strip().decode('utf-8').split()
        for i in xrange(int(num)):
            line = fin.readline().strip()
            yield line
    # for line in open('sentence.txt'):
    #     yield line
    pass

def dealSuccess2():
	fout = open('senplus2.txt','w')
	fail = open('fail.txt','w')
	for line in read_sentences('senplus.txt'):
		line = line.strip()
		if line == '欢迎您来创建，与广大网友分享关于该词条的信息。':
			continue
		if re.match('百度百科尚未收录词条.*',line):
			fail.write(line.decode('utf-8')[12:-1].encode('utf-8') + '\n')
			fail.flush()
			continue
		fout.write(line + '\n')
		fout.flush()
	fail.close()
	fout.close()
	pass

def dealSuccess():
	fin = open('sen.txt','r')
	fout = open('sen2.txt','w')
	fail = open('fail.txt','w')
	while True:
		line = fin.readline()
		if len(line)==0:
			break
		line = line.strip()
		if line == '欢迎您来创建，与广大网友分享关于该词条的信息。':
			continue
		if re.match('百度百科尚未收录词条.*',line):
			#fout.write(line + '\n')
			fail.write(line.decode('utf-8')[12:-1].encode('utf-8') + '\n')
			fail.flush()
			continue
		fout.write(line + '\n')
		fout.flush()
	fin.close()
	fail.close()
	fout.close()
	pass



dealSuccess2()
