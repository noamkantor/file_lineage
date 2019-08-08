############### VALUE MATCHING ###############
###MAIN FUNCTIONS: prepare_preloaded_frequency_dictionary, search_uncommon_vals###
###PRIMARY NOTES###
#this could easily be made to work on an entire directory at a time.
#can be updated as needed with more metadata, since as far as I can tell only around 5% of variables (with repetition) have catalogued most frequent values
#also this currently only works on sas7bdat files. that updating is done with the functions retrieve and create_new_frequency_dictionary.


###USAGE EXAMPLES###
#this one is pretty self-explanatory!

def test():
    prepare_frequency_dictionary()
    c = search_uncommon_vals(r'/adult_analysis_data.dta', r'/some_directory', docfreq_limit=3, time_limit=10)
    for x in c[0]:
        print(x, '\n')


###########


import sqlite3
from pprint import pprint
import csv
import pickle
from tqdm.autonotebook import tqdm
from collections import Counter
import re
from sas_utils import sas2csv
import os
import itertools
import pandas as pd
import sys
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from numpy import array
from time import time
import math
import random

import sys
from collections import OrderedDict

#the next two functions are just documentation of how I created the files 'freq_vals_list.pickle' and ordict_inverted that are used as defaults in the value matching programs.

#do not run this function unless you want to reload a large amount of metadata and write over the file 'freq_vals_list.pickle,' which runs the whole frequency calculation process below
def retrieve(metadata_file, output_file):
  db = sqlite3.connect(metadata_file)
  cursor = db.cursor()


  select_command = '''
      SELECT
      *
      FROM
      varsView;
      '''
  cursor.execute(select_command)
  colnames = [d[0] for d in cursor.description]
  listOfVars = []
  for row in cursor:
    varDict = dict(zip(colnames, row))
    listOfVars.append([varDict['varName'], varDict['varLabel'], varDict['filePath'], varDict['varFreqVals']])
  pprint(listOfVars[1:10])
  
  with open(output_file, 'wb') as f: #output_file was originally 'freq_vals_list.pickle'
      pickle.dump(listOfVars, f)
  return listOfVars


def create_new_frequency_dictionary():
    with open('freq_vals_list.pickle', 'rb') as f:
        l = pickle.load(f)
    
    list_of_vals = [x[3].split('"') for x in l]
    
    final_list_vals = OrderedDict([])
    for index, var in tqdm(enumerate(list_of_vals)):
        final_list_vals[index] = []
        for i in range(sys.maxsize**10):
            try:
                final_list_vals[index].append(var[1+4*i])
            except:
                break
    
    for i in range(len(l)):
        l[i][3] = final_list_vals[i]
    
    
    names, labels, paths, values = zip(*l)
    values = pd.Series(values)
    
    freq_df = pd.DataFrame(columns = ['name', 'label', 'path', 'value'])
    freq_df['name'] = names
    freq_df['label'] = labels
    freq_df['path'] = paths
    freq_df['value'] = values
    
    
    tuples = []
    for x in freq_df.itertuples():
        for m in x.value:
            tuples.append([x.path,m])
    
    more_paths, more_values = zip(*tuples)
    expanded_df = pd.DataFrame(columns = ['path', 'value'])
    expanded_df['path'] = more_paths
    expanded_df['value'] = more_values
    
    numerics = expanded_df.groupby(['value']).path.nunique()
    
    inverted_df = expanded_df.groupby(['value']).path.apply(set)
    
    global ordict_inverted
    ordict_inverted = OrderedDict([])
    for index, value in tqdm(inverted_df.items()):
        ordict_inverted[index] = value
    
    ordict_inverted = OrderedDict(sorted(ordict_inverted.items(), key = lambda x: len(x[1]))) #saves a global ordered dictionary called ordict_inverted that has: as keys, values taken in the metadata,
    #and as values, the files containing those values. the dictionary is ordered by document frequency, so that the values which appear in the fewest files appear first.


def prepare_preloaded_frequency_dictionary():
    with open('labeled_freqs.pickle', 'rb') as f:
        global inverted_df
        inverted_df = pickle.load(f)

    global ordict_inverted    
    ordict_inverted = OrderedDict([])
    for index, value in tqdm(inverted_df.items()):
        ordict_inverted[index] = value

    ordict_inverted = OrderedDict(sorted(ordict_inverted.items(), key = lambda x: len(x[1])))
    
    

def search_uncommon_vals(infile, tempdir, freqs = ordict_inverted, time_limit = 30, docfreq_limit=5):
    docfreq_calculation = 0
    for k,v in freqs.items():
        if len(v)<docfreq_limit:
            docfreq_calculation +=1
        else:
            break
    l = list(freqs.keys())[:docfreq_calculation]
    
    
    start_total = time()
    parents = []
    if infile.endswith('.sas7bdat'):
        temppath = tempdir + r'/temp_freq.csv'
        sas2csv(infile, temppath, tempdir)
        print('sas file converted...')
        rdr = csv.reader(open(temppath, 'r'))
        print('reader initiated...')
        header = next(rdr)
        print(header)
        
        start = time()
        for row in tqdm(rdr):
            if time() - start < time_limit:
                for k, value in enumerate(row):
                    try:
                        int(value)
                        #if len(value) != 0: 
                        if len(value) != 9 and len(value) != 21 and \
                        len(value) != 0 and int(value)>500 and \
                        ('naics' not in header[k]) and ('year' not in header[k]):
                            #removes pik and carra uid
                            #print('looking for value:', value, 'in', limit, 'least frequent values in corpus')
                            if value in l: #need to figure out a way to put limit in here
                                parents += freqs[value]
                                parents_with_values[value] = freqs[value]
                                #if any(['cms' == x.split(r'/')[2] for x in freqs[value]]):
                                    #print('cms:', header[k], value)
                                #if any(['state' == x.split(r'/')[2] for x in freqs[value]]):
                                    #print('state:', header[k], value)
                    except:
                        pass
            else:
                break
        
        
        
        
        
    elif infile.endswith('.dta'):
        rdr = pd.read_stata(infile, iterator=True)
        #try to look at random place.
        #if dataset has less than 100 rows, we just start at beginning
        rand = random.randint(0,100)
        try: 
            for i in range(rand):
                next(rdr)
        except:
            rand = 0
    
        start = time()
        for k, row in tqdm(enumerate(rdr, rand)):
            if time() - start < time_limit:
                for column in row:
                    value = row.loc[k,column]
                    try:
                        int(value)
                        #if len(value) != 0: 
                        #if True:
                        if len(str(value)) != 9 and \
                        len(str(value)) != 21 and \
                        len(str(value)) != 0 and \
                        int(value)>500 and \
                        ('naics' not in column) and \
                        ('year' not in column) and \
                        ('YYYY' not in column): #removes pik and carra uid
                            if str(value) in l: 
                                parents += freqs[str(value)]

                    except:
                        pass
            else:
                break
    
    if infile.endswith('.sas7bdat'):
        os.remove(temppath)
    
    directories = [x.split(r'/')[0] + r'/' + x.split(r'/')[1] + r'/' + x.split(r'/')[2] + r'/' + x.split(r'/')[3] for x in parents]
    
    
    print('Total time:', time()-start_total)
    return Counter(parents).most_common(), Counter(directories)



