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
import jieba
reload(sys)

def read_sentences(_file = 'sen.txt'):
    fin = open(_file,'r')
    while True:
        line = fin.readline()
        if len(line) == 0:
            break
        key,num = line.strip().decode('utf-8').split()
        for i in xrange(int(num)):
            line = fin.readline().strip().decode('utf-8')
            yield line
    # for line in open('sentence.txt'):
    #     yield line
    pass

def check_all_chinese(check_str):
    if isinstance(check_str, unicode):
        for ch in check_str:
            if u'\u4e00' <= ch <= u'\u9fff':
                continue
            else:
                return False
    else:
        for ch in check_str.decode('utf-8'):
            if u'\u4e00' <= ch <= u'\u9fff':
                continue
            else:
                return False
    return True

def ltp_segment():
    url = 'http://api.ltp-cloud.com/analysis/'
    param = {'api_key':'n5O4a8c7TpMJRxQYwdOGIJJPdJHBfd6Y9NnhNXaT',
             'text':'',
             'pattern':'pos',
             'format':'plain'
             }
    omits = ['wp','_m','ws']
    fout = open('segedsentence.txt','w')
    num = 0
    for s in read_sentences():
        num += 1
        print '%6d'%(num),
        param['text'] = s
        r = requests.get(url,params = param).content.split()
        tmps = ''
        for i in r:
            if len(i)<=2 or i[-2:] in omits:
                continue
            i = i[:i.rfind('_')]
            tmps += i + ' '
        tmps = tmps[:-1]
        fout.write(tmps + '\n')
        print tmps
    pass

def jieba_segment():
    num = 0
    fout = open('segedsentence.txt','w')
    for s in read_sentences():
        num += 1
        print '%6d'%(num)
        seg_iter = jieba.cut(s,cut_all = False)
        tmps = ''
        for i in seg_iter:
            if check_all_chinese(i):
                tmps += i + ' '
        tmps = tmps[:-1]
        fout.write(tmps.encode('utf-8') + '\n')
        fout.flush()
        #print tmps.encode('utf-8')
    fout.close()
    pass

def __main__():
    # ltp_segment()
    jieba_segment()
    pass


__main__()

