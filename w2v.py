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
import gensim, logging
reload(sys)

class MySentences(object):
	def __init__(self):
		pass
	def __iter__(self):
		for line in open('segedsentence.txt'):
			yield line.split()
		pass


def w2v():
	sentences = MySentences()
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	print '#############\nw2v start:'
	model = gensim.models.Word2Vec(sentences, min_count=5, size = 100, workers = 2)
	print '\n#############\nw2v end'
	model.save('model.w2v')
	print 'w2v saved'
	pass


w2v()