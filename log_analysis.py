############### LOG ANALYSIS ###############
###MAIN FUNCTIOSN: approximateLineage (starts with child and looks upward), initiate_graph (simple graph instance for use with update_graph), update_graph (starts with parent and looks downward)###
###PRIMARY NOTES###
#there are two ways in which you might want to change this package. first of all, it computes weights based on how far away things are from each other in the log. while theoretically cool (and reflective of a disease-based model of file commingling), this might be computationally expensive and could be dispensed with. in the same vein, I have put more attributes that can be assigned to an edge, like year matches. they can also be removed if needed. i have also integrated variable simmilarity into this module. since variable similarity takes such a long time, i think this shoud ultimately be removed and replaced at most by variable similarity with the package fuzzywuzzy or no variable similarity at all, or made optional. in fact, that is a near-term goal of mine.


#the second way in which this package could be modified would be to run it for every file on a log, not just a fixed child file or parent file. the way to do this would just be to iterate approximateLineage along all file names in the log. it is likely that approximateLineage could be optimized so that this iteration doesn't repeat any unnecessary steps, but I haven't done that yet.


#I wrote the parent-down algorithms in numpy and the child-up in pandas, with the hunch that the numpy would be faster. They can both be changed without much work to do either.

###USAGE EXAMPLE 1###
#the output of this example can be found on my final slides for OPCOM. it uses the fake logs listed.

def test1():
    year = pd.Timedelta('365 days')
    day = pd.Timedelta('1 days')
    hour = pd.Timedelta('1 hours')
    minute = pd.Timedelta('1 minutes')
    second = pd.Timedelta('1 seconds')
    answer = approximateLineage(r'M:\fake_sftp_info.csv',
                                r'M:\fake_child.csv',
                                .01000, 10*minute,
                                year, True)
    
    plt.figure(1)
    plt.axes()
    pos = nx.circular_layout(answer[0])
    
        
    colors = []
    for u,v in answer[0].edges():
            colors.append(answer[0][u][v]['color'])
    
    print(colors)
    
    nx.draw(answer[0], pos, with_labels=False, edge_color=colors)
    plt.show()

### USAGE EXAMPLE 2 ###

def test2():
    g = initiate_graph(r'\fake_parent.csv')
    p= update_graph(r'\fake_sftp_info.csv', r'\fake_parent.csv', np.timedelta64(300000,'s'), g)
    
    nx.draw(p,pos=nx.circular_layout(p), with_labels=True)

###########



#the goal of this code is to analyze the splunk code coming from logs,
#and build approximate lineage and association trees
import pandas as pd
import datetime
import math
import networkx as nx
import re
import matplotlib.pyplot as plt
#from fuzzywuzzy import fuzz
#from fuzzywuzzy import process
#import pygraphviz as pgv
import csv
#import sklearn
import numpy as np
from numpy import genfromtxt
import pprint
#import pickle
import ntpath
import sqlite3

from gensim.models.wrappers import FastText
from gensim.models import KeyedVectors
from gensim.similarities import WmdSimilarity
from decimal import Decimal
from decimal import ROUND_UP
from nltk.corpus import stopwords

from nltk import download
download('stopwords')
stop_words = stopwords.words('english')


def split_nonempty(str):
    #splits a string along non-alphanumeric characters
    sentence = re.split('[^a-zA-Z0-9]', str.lower())
    return [y for y in sentence if (y!='' and y not in stop_words)]


def commonYears(string1, string2):
    #returns a common year value between 1900 and 2039 if both strings share that value, otherwise returns none
    exp = re.compile('19[0-9][0-9]|20[0-3][0-9]')
    y1 = exp.search(string1)
    y2 = exp.search(string2)
    if y1 and y2:
        return y1.group() == y2.group()

def isBoolFlStr(val):
    #returns the type of a float or string
    try:
        return float(val)
    except TypeError:
        try:
            return str(val)
        except TypeError:
            return None

        
def sameType(val1, val2):
    if isBoolFlStr(val1) == None or isBoolFlStr(val2) == None:
        print('Can only compare Boolean, float, and string types.')
    else:
        return type(isBoolFlStr(val1)) == type(isBoolFlStr(val1))       
        
        
def get_vars(path, var_loc): #var_loc can be either 'column' or 'row'    
    #pulls list of variable names out of either column or row of dataset
    vars = []
    if nameext(path)[1] == 'csv':
        rdr = csv.reader(open(path)) #or if not -- csv, whatever it is. might just want this to be a list
        if var_loc == 'column':
            for line in rdr:
                vars.append(line[0])
        if var_loc == 'row':
            vars = next(rdr)
    
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


def parentWeights(df, filepath, sigma, cutoff):
    #takes a log csv file and the filepath of which we want
    #the lineage, the decay rate sigma for the weighting,
    #and a cutoff time after which we don't consider two files to be related
    
    #picks out all files that were accessed shortly before 'filepath' first appears in the log


    df_child = df.loc[df['name'] == filepath]
    birth = df_child.head(1)
    creator = birth.iloc[0]['jbid']
    list = birth.index.values.tolist()
    birth_time = birth.iloc[0]['Time']
    earlierFiles = df.loc[(df['Time'] < birth_time) &
                          (abs(df['Time'] - birth_time) < cutoff) &
                          (df['jbid'] == creator) & (df['key'] == 'access')]
    earlierWeighted = earlierFiles[['name']]
    #now add weights
    list = []
    for index, row in earlierFiles.iterrows():
        list.append(math.exp(-1.0*(pd.Timedelta.total_seconds(
                 df.iloc[index]['Time'] - birth_time)/10000.0)**2/(2.0*sigma)))
    earlierWeighted = earlierWeighted.assign(weight = list)
    return earlierWeighted
    #closer files are weighted higher

#the following method builds a network of nodes, each corresponding to a file
#in the log. The edges and weights in the network come from the parentWeights method.
#Additionally, each of the edges is given several attributes from our toolbox.
def lineage_graph(df, filepath, sigma, cutoff, graph):
    nextWeights = parentWeights(df, filepath, sigma, cutoff)
    if nextWeights.shape[0] != 0:
        for index,row in nextWeights.iterrows():
            graph.add_node(row['name'])
            
            
            
            year_match = commonYears(filepath,row['name'])
            
            #here are possible edge attributes coming from other lineage measures.
            #currently commented out.
            
            #code_connections = False
            
            
            #numerical_matches = False
            
            
            #coappearance = 0
            
            #file_name_sim = fuzz.partial_ratio(filepath, row['name'])
            
            
            sim_scores = {'year_match': year_match,
                          #'code_connections': code_connections,
                          #'numerical_matches': numerical_matches,
                          #'coappearance': coappearance,
                          'color': 'b'
                          }
            
            
            
            graph.add_edge(filepath, row['name'], 
                           weight = Decimal(row['weight']).quantize(Decimal('.0001'), rounding=ROUND_UP), 
                            year_match = year_match,
                          #code_connections = code_connections,
                          #numerical_matches = numerical_matches,
                          #coappearance = coappearance,
                          color = 'b')
            #now we iterate
            lineage_graph(df, row['name'], sigma, cutoff, graph)
    return graph


def approximateLineage(log, filepath, sigma, cutoff_birth, cutoff_acq, all_acq):
    #requires a csv log file, a scaling factor sigma, a filepath
    #to start with in the search, a cutoff time, and a boolean that
    #says whether we want to examine the whole log for files that
    #appeared near each other, not only near each others' births. all_acq calls addAllAcq.
    
    #returns a built lineage graph, using lineage_graph and parent_weights, as well as
    #the files that have in edges but no out edges. These are ccalled sources and considered
    #possible parent files.
    
    G = nx.DiGraph()
    df = pd.read_excel(log)
    #now I experimented with labelling the initial node with some data, specifically IRE
    #variable matches and the variable FTI estimation. But might as well just do the sources.
    #variable_FTI_estim = 0.714
            
            
    #IRE_variable_matches = 2 #['sample variable'] I want this to be a list!


    G.add_node(filepath)
    #nx.set_node_attributes(G, 'variable_FTI_estim', variable_FTI_estim)
    #nx.set_node_attributes(G, 'IRE_variable_matches', IRE_variable_matches)

    graph = lineage_graph(df, filepath, sigma, cutoff_birth, G)
    if all_acq:
        addAllAcqu(graph, df, filepath, cutoff_acq)
    sources = [x for x in graph.nodes if graph.out_degree(x)==0]
    #this needs to be changed: if acquaintance edges are added, the source files
    
    
    
    return graph, sources

def addAllAcqu(graph, df, filepath, cutoff):
    #this function adds acquaintance data, explained below, to an existing lineage graph
    
    #df is the imported log, graph is the lineage graph we'd like to add
    #"acquaintance" data to, meaning we consider two files acquaintances if they
    #appear near each other in the log. filepath is the path of the file whose lineage we would like
    #and cutoff is the time gap after which we don't consider two files to be acquaintances
    
    #by initializing this function with an empty DiGraph and a PARENT file, we can build a potentially
    #huge acquaintance graph
    new_nodes = []
    df_child = df.loc[df['name'] == filepath]
    #looks for all files opened by the same user and after the file within the cutoff period
    for event1 in df_child.itertuples():
        for event2 in (df.iloc[event1.Index:,:]).itertuples():
            if abs(event2.Time - event1.Time) < cutoff:
                if event2.jbid == event1.jbid: #should change this to require the key to be access
                    if event2.name in list(graph.nodes) and event2.name != event1.name:
                        if (event1.name, event2.name) not in graph.edges() and (event2.name, event1.name) not in graph.edges():
                            #nx.set_node_attributes(graph, {event2.name: {'also_acquaintance': True}}) #also_acq is stupid
                            graph.add_edge(event1.name, event2.name)
                            nx.set_edge_attributes(graph,{(event1.name,event2.name):{'color': 'r'}})
                            graph.add_edge(event2.name, event1.name)
                            nx.set_edge_attributes(graph,{(event2.name,event1.name):{'color': 'r'}})
                    elif event2.name != event1.name:
                        new_nodes.append(event2.name)
                        graph.add_node(event2.name)
                        nx.set_node_attributes(graph,{event2.name: {'acq': True}})
                        graph.add_edge(event1.name, event2.name)
                        nx.set_edge_attributes(graph,{(event1.name,event2.name):{'color': 'r'}})
                        graph.add_edge(event2.name, event1.name)
                        nx.set_edge_attributes(graph,{(event2.name,event1.name):{'color': 'r'}})
            else:
                break
    #looks for nearby files that were opened before the file -- just reverse of previous           
    for event1 in df_child.itertuples():
        for event2 in df[::-1].iloc[event1.Index:,:].itertuples():
            if abs(event2.Time - event1.Time) < cutoff:
                if event2.jbid == event1.jbid:
                    if event2.name in list(graph.nodes) and event2.name != event1.name:
                        if (event1.name, event2.name) not in graph.edges() and (event2.name, event1.name) not in graph.edges():
                            #nx.set_node_attributes(graph, {event2.name: {'also_acquaintance': True}})
                            graph.add_edge(event1.name, event2.name)
                            nx.set_edge_attributes(graph,{(event1.name,event2.name):{'color': 'r'}})
                            graph.add_edge(event2.name, event1.name)
                            nx.set_edge_attributes(graph,{(event2.name,event1.name):{'color': 'r'}})
                    elif event2.name != event1.name:
                        new_nodes.append(event2.name)
                        graph.add_node(event2.name)
                        nx.set_node_attributes(graph,{event2.name: {'acq': True}})
                        graph.add_edge(event1.name, event2.name)
                        nx.set_edge_attributes(graph,{(event1.name,event2.name):{'color': 'r'}})
                        graph.add_edge(event2.name, event1.name)
                        nx.set_edge_attributes(graph,{(event2.name,event1.name):{'color': 'r'}})
            else:
                break
    print(len(new_nodes))
    if len(new_nodes) > 0:
        for relative in new_nodes:
            addAllAcqu(graph,df,relative, cutoff)
    else:
        return
    
    
def find_children_broadcasting(log_file, parent, time_limit=60*60, include_acq=False):
    #time_limit should eventually be replaced by the time when the user logs out
    #currently is unidirectional, ie. matters which file is opened first.
    #have to create another time_ranges column if I want to fix that.
    #since the log only has minute-by-minute resolution, this gives ties a doubled edge
    
    children = []
    acq = []
    
    #convert into the datetime format we need
    #if final log ends up lookign different from this, will have to change by hand the astype statements below
    #that I used to increase allowed string length and change to datetime formats
    log_array = np.genfromtxt(log_file, delimiter=',', dtype=None, names=True, encoding=None, usecols=('Time', 'jbid', 'cwd', 'exe', 'name', 'key'))
    log_array = log_array.astype([('Time', '<U25'), ('jbid', '<U8'), ('cwd', '<U16'), ('exe', '<U32'), ('name', '<U86'), ('key', '<U6')])

    for i in range(log_array.shape[0]):
        time = log_array[i]['Time'].split(' ')[1]
        if int(time.split(':')[0]) < 10:
            time = '0' + time
        calendar = log_array[i]['Time'].split(' ')[0]
        year = calendar.split('/')[2]
        month = calendar.split('/')[0]
        if int(month) < 10:
            month = '0' + month
        day = calendar.split('/')[1]
        if int(day) < 10:
            day = '0' + day
        
        log_array[i]['Time'] = year + '-' + month + '-' + day + 'T' + time
        log_array[i]['Time'] = np.datetime64(log_array[i]['Time'])
    log_array = log_array.astype([('Time', 'datetime64[s]'), ('jbid', '<U8'), ('cwd', '<U16'), ('exe', '<U32'), ('name', '<U86'), ('key', '<U6')])
    
    #this will store, for each user, the relevant time frames after opening of parent
    time_ranges = np.zeros((log_array.shape[0],), \
                           dtype=[('t1','datetime64[s]'), ('t2','datetime64[s]'), ('jbid','<U8')])
    time_ranges['t1'] = log_array['Time'] 
    time_ranges['t2'] = log_array['Time'] + time_limit
    time_ranges['jbid'] = log_array['jbid']
    
    parent_appearances = log_array[np.where(np.logical_and(log_array['name'] == parent, log_array['key'] == 'access'))]
    parents_with_ranges = np.zeros(parent_appearances.shape, dtype = parent_appearances.dtype.descr + [('t2', 'datetime64[s]')])

    parents_with_ranges['Time'] = parent_appearances['Time']
    parents_with_ranges['jbid'] = parent_appearances['jbid']
    parents_with_ranges['cwd'] = parent_appearances['cwd']
    parents_with_ranges['exe'] = parent_appearances['exe']
    parents_with_ranges['name'] = parent_appearances['name']
    parents_with_ranges['key'] = parent_appearances['key']
    parents_with_ranges['t2'] = parent_appearances['Time'] + time_limit
    
    #still need to integrate the fact that we care about this for different users
    users = np.unique(parents_with_ranges['jbid'])
    #we could slice by user and then iterate along those slices. even better: broadcast along all users
    
    
    #can either look at acquantainces or just openings before birth
    for u in users:
        if include_acq:
            #slice for users
            temp = parents_with_ranges[parents_with_ranges['jbid'] == u]
            
            #reshape for broadcasting
            resh1 = temp['Time'].reshape((parents_with_ranges['Time'].shape[0],1))
            resh2 = temp['t2'].reshape((parents_with_ranges['t2'].shape[0],1))

            #slice for users
            log_array_u = log_array[log_array['jbid'] == u]

            g = np.greater_equal(log_array_u['Time'] , resh1)
            l = np.less(log_array_u['Time'], resh2)
            a = np.logical_and(g,l)
            o = np.logical_or.reduce(a)

            b = log_array[o]
            acq += list(set(b['name']))
            acq.remove(parent)
        
        #now restrict to just births
        birth_indices = np.unique(log_array['name'], return_index=True)[1]
        births = np.take(log_array, birth_indices)
        
        temp = parents_with_ranges[parents_with_ranges['jbid'] == u]

        resh1 = temp['Time'].reshape((parents_with_ranges['Time'].shape[0],1))
        resh2 = temp['t2'].reshape((parents_with_ranges['t2'].shape[0],1))

        births_u = births[births['jbid'] == u]

        #look at events that happen between time of opening of parent and time_limit later
        g = np.greater_equal(births_u['Time'] , resh1)
        l = np.less(births_u['Time'], resh2)
        a = np.logical_and(g,l)
        o = np.logical_or.reduce(a)

        b = births_u[o]
        children += list(set(b['name']))
        children.remove(parent)
    
    if include_acq:
        return children, acq
    else:
        return children
        
    
    
def initiate_graph(parent):
    g = nx.DiGraph()
    g.add_node(parent)
    return g

def update_graph(log_file, parent, time_limit, graph):
    for x in find_children_broadcasting(log_file, parent, time_limit):
        if x not in graph:
            graph.add_node(x)
            update_graph(log_file, x, time_limit, graph)
        if (parent,x) not in graph.edges:
            graph.add_edge(parent,x)
        
    return graph

#maybe also let us start part way down the log by chopping off the first part at a specified index    
    
    

