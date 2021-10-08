#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 13:48:56 2021

@author: janeyalex
"""
import pandas as pd
import collections


f = open("ulysses.txt",'r')
l1=[]
l2=[]
for line in f:
    lineword=line.split(':')
    #if not lineword[0].isdigit():
    l1.append(lineword[0])
    l2.append(int(lineword[1]))          
f.close()

df = pd.DataFrame({'word': l1,'count':l2})

# 5a
totalNumWords = df['count'].sum()
uniqueWords =len(l2)
frac = uniqueWords/totalNumWords

freq= collections.Counter(l2)

# 5b
n1=freq[1]/sum(freq.values())
n2=freq[2]/sum(freq.values())
n3=freq[3]/sum(freq.values())