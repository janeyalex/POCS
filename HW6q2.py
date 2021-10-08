import numpy as np
from scipy import stats
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import collections
import urllib.request
from string import punctuation




def question_two():

    def simons_model(steps, rho):
        population = np.zeros(steps)
        num_groups = 1
        population[0] = 1

        for i in range(steps - 1):
            if np.random.uniform(0,1) <= rho:
                num_groups += 1
                population[i + 1] = num_groups
            else:
                population[i + 1] = np.random.choice(population[0 : i + 1])

        return population

    rhos = [0.1, 0.01, 0.001]
    for rho in rhos:
        times = 10
        steps = 100000
        res = np.empty([times, steps])

        for i in range(times):
            res[i, :] = simons_model(steps, rho)

        # find counts of each group
        unique, counts = np.unique(res, return_counts=True)
        unique = unique.astype(int)
        counts = sorted(counts)[::-1]

        frequency = np.log10(counts)
        rank = np.log10([i + 1 for i in range(len(counts))])

        regr = stats.linregress(rank[1:], frequency[1:])
        beta = -regr.slope
        alpha = 1 - rho

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rank, y=frequency, name='Zipf', mode='markers'))
        fig.add_trace(go.Scatter(x=rank, y=regr.intercept + rank*regr.slope, name='Regression Fit', mode='lines'))
        fig.update_xaxes(title='log rank')
        fig.update_yaxes(title='log frequency')
        fig.update_layout(title=f'Zipfian Distribution for Simon Model for {steps} steps and rho={rho}. alpha={alpha:.3f}, beta={beta:.3f}')
        fig.write_image(f'{rho}.png', width=1000, height=800, format='png')
        
def question_six():
    def words_of_book():
        """Download `A tale of two cities` from Project Gutenberg. Return a list of
        words. Punctuation has been removed and upper-case letters have been
        replaced with lower-case.
        """
        #check to see if book has already been downloaded
        try:
            #if file is found, open file and read it
            f = open("pride.txt",'r')
            raw = f.read()
        
        #if file is not found download book
        except FileNotFoundError:
            # DOWNLOAD BOOK:
            url = "https://www.gutenberg.org/cache/epub/42671/pg42671.txt"
            req = urllib.request.urlopen(url)
            charset = req.headers.get_content_charset()
            if charset is None:
                charset = 'utf-8'
            raw = req.read().decode(charset)
            #write decoded text to file 
            with open("pride.txt",'w') as f:
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
        return list_words
    wordList= words_of_book()
#question_two()
question_six()

