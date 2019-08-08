############### VARIABLE MATCHING ###############
###MAIN FUNCTIONS: initiate_variable_matching, load_facebook_model, load_metadata_model, letsquery###
###PRIMARY NOTES###
#the primary drawback of this module is that it is SLOW. I wasn't able to get to this, but I have a feeling that the module fuzzywuzzy would do just as well. this could also be parallelized. Because it is so slow, I chose not to write it in a way that it could search whole directories, but it could be easily modified to do that. also, could probably make some of the variables defaulted because letsquery is a little... verbose right now

###USAGE EXAMPLES###

#run initiate variable matching before these

#a simple example:

def test1():
    return letsquery('some_dataset.csv',
                       model_fb, labels_split, 10, 'row', df, 1, df, 0.55)



#a more complicated example, where we only want to test for variables that are in the economic directory
def test2():
    filtered_by_e = filter_by_directory(df, 'economics')
    
    labels_split_e_only = []
    for x in filtered_by_e[1]:
        newsentence = split_nonempty(x)
        if (len(newsentence) != 0):
            labels_split_e_only.append(newsentence)
            if len(labels_split_e_only) % 100 == 0:
                print(len(labels_split_e_only))
                print(newsentence)
    
    return letsquery('Modified_variable_names.csv',
                       model_fb, labels_split_e_only, 10, 'column', df, 1, filtered_by_e[0], 0.55)




###########




from gensim.models import FastText
from gensim.similarities import WmdSimilarity
from nltk.corpus import stopwords
from nltk import download
download('stopwords')
stop_words = stopwords.words('english')
#in the presence of a firewall, you can load by hand the list of stopwords in stopword_text in the files_needed_for_lineage_tools_final folder.
import pandas as pd
import re
import pickle
import numpy as np
import pprint
import os
from time import time
start_nb = time()
import csv
from collections import Counter
import ntpath
import sas7bdat
from sas7bdat import SAS7BDAT
import sqlite3
from dbfread import DBF
from tqdm.autonotebook import tqdm
from gensim.test.utils import datapath

def split_nonempty(str):
    #splits a string along non-alphanumeric characters
    sentence = re.split('[^a-zA-Z0-9]', str.lower())
    return [y for y in sentence if (y!='' and y not in stop_words)]

def respace(l):
    #puts together a string that has been split with the last method, spaces in between
    return ' '.join(l)

def get_vars(path, var_loc): #var_loc can be either 'column' or 'row'    
    #pulls list of variable names out of either column or row of dataset
    vars = []
    if nameext(path)[1] == 'csv':
        rdr = csv.reader(open(path)) #or if not -- csv, whatever it is. might just want this to be a list
        if var_loc == 'column':
            for line in rdr:
                vars.append(line[0])
        if var_loc == 'row':
            vars = rdr.next()
    
    elif nameext(path)[1] == 'dta':
        rdr = read_stata(open(path))
        vars = list((rdr.variable_labels()).values())
        
    elif nameext(path)[1] == 'sas7bdat':
        with SAS7BDAT(path) as rdr: #idk whether I need to open the path here
            for col in rdr.columns:
                vars.append((col.label).decode('utf-8'))
        vars = list(filter(lambda x: x != 'Missing Value', vars))
    
    elif nameext(path)[1] == 'sqlite':
        db = sqlite3.connect(path)
        all_tables = []
        main_cursor = db.cursor()
        main_cursor.execute('SELECT name FROM sqlite_master WHERE type=\'table\'')
        for row in cursor:
            all_tables.append(row[0])
        main_cursor.close()
            
        for x in all_tables:
            cursor = db.cursor()
            cursor.execute('SELECT * FROM pragma_table_info(' + x + ')')
            for row in cursor:
                vars.append(row[1])
            db.close()
    
    elif nameext(path)[1] == 'dbf':
        vars = DBF(path).field_names

    else:
        print('Your file type is not currently supported.')
        
    return vars


def filter_by_directory(df, *args):
    #return df[ x in df['path'] for x in args]
    
    filtered = {}
    for i in args:
        filtered[i] = df[ df['path'].str.contains(i)]
    x = pd.concat(filtered)
    return x, x['label'].unique().tolist() + x['name'].unique().tolist()


def nameext(path):
    #takes a path in the form of a string and splits it into file type and everything before file type
    # (i.e., the "preextension")
    head, tail = ntpath.split(path)
    full = tail or ntppath.basename(head)
    split = full.split('.')
    #splits filename using periods
    preext = ''
    for i, x in enumerate(split):
        if i < len(split)-2:
            preext += x + '.'
        if i == len(split)-2:
            preext += x
    #groups together all words with periods before final one, i.e. 'text.alpha.pdf' will have preext equal to 'text.alpha'
    return   preext, split[len(split)-1]
#the above code gives filename and the extension. the concatenation has to occur because there may be periods in the file name

def initiate_variable_matching():
    #this script loads a list of lists of length 3. Each list contains a variable name,
    #variable label, and path to the file containing that variable in the metadata

    print('loading list of vars, labels, and paths...')
    with open('vars_names_files.pickle','rb') as f:
        global list1
        list1 = pickle.load(f)

    print('making a dataframe')
    global df
    df = pd.DataFrame(list1, columns = ['name', 'label', 'path'], dtype=object)

    #create labels that are split according to FastText's format for input corpus
    df['split_label'] = np.nan
    df['split_label'] = df['split_label'].astype(object)
    df['spaced_label'] = np.nan
    df['spaced_label'] = df['spaced_label'].astype(object)

    print('preprocessing data for model use')
    for row in df.itertuples():
        df.iat[row.Index, 3] = split_nonempty(row.label)
        #df.iloc[i][df.columns.get_loc('split_label')] = split_nonempty(str(row['label']))
        df.iat[row.Index, 4] = respace(df.iat[row.Index,3])

    #take out repeated labels. surprisingly hard to figure out how
    #best to do this, since .unique() can't be used on lists of lists --
    #it only works on hashable objects.
    global labels_norep
    labels_norep = df.loc[:,'label'].unique()
    print('There are ' + str(labels_norep.size) + ' distinct labels.')

    global labels_rep
    labels_rep = df.loc[:,'label']

    #now we finally make a list of split up labels that we are gonna use for the model training
    global labels_split
    labels_split = []

    for x in labels_norep.tolist():
        newsentence = split_nonempty(x)
        if (len(newsentence) != 0):
            labels_split.append(newsentence)
            if len(labels_split) % 10000 == 0:
                print(len(labels_split))
                print(newsentence)

                
def train_model():
    #this trains our fasttext model
    global model_labels
    model_labels = FastText(window=3, word_ngrams=1)
    model_labels.build_vocab(sentences=labels_split)
    model_labels.train(sentences = labels_split, total_examples=len(labels_split), epochs=10)


def load_facebook_model():
    #I think this datapath function works with relative paths, if not you might have to do it by hand with the full path
    #of wherever the cc.en.300.bin file is
    cap_path = datapath('cc.en.300.bin')
    global model_fb
    model_fb = FastText.load_fasttext_format(cap_path)

def load_metadata_model():
    #this loads our pretrained model.
    #this model has been trained without stopwords and without repeated labels
    model_labels = FastText.load('FT_no_stop_words.gz')
    
def letsquery(testfile, model, corpus, n_output, location, dataframe, n_varmatches, restricted_df, threshold): 
    #testfile is the file whose variables we want to understand, model is the
    #pretrained word2vec or FastText model, corpus is the set of labels or variables we would like to query from,
    #n_varmatches is the number of high-scoring variables to which we limit ourselves for each mystery variable.
    #location is either 'column' or 'row'.
    #depending on whether testfile has variables as columns or rows.
    #dataframe is master dataframe containing file names, labels, and paths.
    #restricted data frame should be a dataframe that only contains variable from econ files, say, if you only
    #want to test those
    
    
    start = time()
    match_files = []
    preintersection = []
    word_list = []
    #pull variables from data set we are investigating
    test_vars = get_vars(testfile, location)
    #print('Attempting to identify the following variables:', test_vars)
    #initiate instance of searching corpus of labels for the test variables
    instance = WmdSimilarity(corpus, model, num_best=n_varmatches)
    
    #find all files that contain the matched variables
    for var in tqdm(test_vars):
        query = split_nonempty(var)
        response = instance[query]
        word_list.append(response[0][0])
        for i in range(n_varmatches):
            if response[i][1] >= threshold:
                slc = restricted_df[restricted_df['spaced_label']  == respace(corpus[response[i][0]])]
                names = slc['path'].unique().tolist()
                match_files += names
            #preintersection.append(set(names))
            #intersection = set.intersection(*preintersection)
    print('I have found ', len(set(match_files)), ' distinct prospective parent files among all variables.')
    #print('I have found ', len(intersection), ' prospective parent files that are common to all variable names.')
    
    #these counters see how often each files appeared in the list of matched files, which
    #includes ALL test variables and ALL prospective match variables for EACH of those test variables
    c = Counter(match_files)
    totals = c.most_common(None)
    mc = totals[:n_output]
    
    #can also split according to which directory the files live in, so that we don't have to look
    #at really long strings
    directories = [directory(x) for x in match_files]
    d = Counter(directories)
    dtotals = d.most_common(None)
    if len(dtotals) >= n_output:
        mcd = dtotals[:n_output]
    else:
        mcd = dtotals
    
    print('The', n_output ,'most common file names are', mc)
    print('The', n_output ,'most common directory names are', mcd)
    print('Total time to process query:', time()-start)
    
    return set(match_files), totals, dtotals, word_list

















