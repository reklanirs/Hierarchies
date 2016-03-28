#!/usr/bin/python2.7
# -*- coding: utf-8 -*- import requests
import os
import re
import sys
import math
import random
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
from sklearn import cluster
import numpy as np
from numbapro import vectorize
from numbapro import cuda
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


model = gensim.models.Word2Vec.load('model/m100.w2v')