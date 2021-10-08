#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 18:37:05 2021

@author: janeyalex
"""

import numpy as np
from scipy import stats
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import collections
import urllib.request
from string import punctuation


def words_of_book(url1,filename):
    
    #check to see if book has already been downloaded
    try:
        #if file is found, open file and read it
        f = open(f'{filename}.txt','r')
        raw = f.read()
    
    #if file is not found download book
    except FileNotFoundError:
        # DOWNLOAD BOOK:
        req = urllib.request.urlopen(url1)
        charset = req.headers.get_content_charset()
        if charset is None:
            charset = 'utf-8'
        raw = req.read().decode(charset)
        #write decoded text to file 
        with open(f'{filename}.txt','w') as f:
            f.write(raw)
                
   
    
    # PARSE BOOK
    raw = raw[750:] # The first 750 or so characters are not part of the book.
    
    # Loop over every character in the string, keep it only if it is NOT
    # punctuation:
    exclude = set(punctuation) # Keep a set of "bad" characters.
    list_letters_noPunct = [ char for char in raw if char not in exclude ]
    
    # Now we have a list of LETTERS, *join* them back together to get words:
    text_noPunct = "".join(list_letters_noPunct)
    # (http://docs.python.org/3/library/stdtypes.html#str.join)
    
    # Split this big string into a list of words:
    list_words = text_noPunct.strip().split()
    
    # Convert to lower-case letters:
    list_words = [ word.lower() for word in list_words ]
    
    f.close()
    
        
    wordFreq= collections.Counter(list_words).items()
    wordFreq=dict(wordFreq)
    l1 =list(wordFreq.keys())
    l2 = list(wordFreq.values())
    
    df = pd.DataFrame({'word': l1,'count':l2})
    
    totalNumWords = df['count'].sum()
    uniqueWords =len(l2)
    frac = uniqueWords/totalNumWords
    print(f"Innovation rate = {frac}")
    
    freq= collections.Counter(l2)
    
    
    n1=freq[1]/sum(freq.values())
    print(f"n1 ={n1}")
    n2=freq[2]/sum(freq.values())
    print(f"n2 ={n2}")
    n3=freq[3]/sum(freq.values())
    print(f"n3 ={n3}")
    return frac

def theoretical(rate):
    n1 = 1/(2-rate)
    n2= (1-rate)/((3-2*rate)*(2-rate))
    n3= (2*(1-rate)**2)/((4-3*rate)*(3-2*rate)*(2-rate))
    return n1,n2,n3
    
prideUrl = "https://www.gutenberg.org/cache/epub/42671/pg42671.txt"
name= "pride"

monteUrl = "https://www.gutenberg.org/files/17989/17989-0.txt"      
name1 = "monte" 


frac1 =words_of_book(prideUrl,name)
frac2 =words_of_book(monteUrl,name1)

n1,n2,n3=theoretical(frac1)
n11,n21,n31=theoretical(frac2)
print(f"Theoretical {n1} {n2} {n3}")
print(f"Theoretical {n11} {n21} {n31}")