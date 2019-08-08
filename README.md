# Tools for identifying file lineage

<img src = "/unlabelled_lineage_ex.png" align="middle">
</img>

This package contains non-sensitive versions of various file lineage tools created for the US
Census Bureau during Summer 2019 as part of the Civic Digital Fellowship.
The general problem: reconstruct file lineage of commingled datasets.

The real challenge with such commingled datasets is that pretty much anything can happen to them.
They can be:
*stripped of multiple columns;
*sliced along a certain subset of rows;
*combined with any number of files along any number of columns;
*modified numerically in arbitrary ways, say by taking the mean along a certain column or using "groupby" methods;
*sorted in arbitrary ways;
*moved to an arbitrary place on a server, with a non-identifying name;
*the names and labels of variables might be changed to something more human-friendly;
or any number of other transformations.

There are three main parts to this module, which encapsulate the tack I decided to take on this problem.

## Value Matching

These algorithms look at a number of values on a SAS or Stata file and tries to guess possible parent files
using known values within Census detasets from the metadata project. Due to obvious
sensitivity constraints, nothing from the metadata is present here. It is included in a
separate file for Census users.

Strength: Relatively fast, turns out there are some values (such as month codes) that would not normally be manipulated. Seems to be quite accurate, at least on a small number of known mixed Census datasets.
Weakness: Requires access to the datasets themselves. Requires extensive pre-computation of frequencies for dataset values
across the corpus of possible parent datasets.

## Log Analysis

These algorithms reconstruct a directed lineage graph (or network) from an input log file.
The reconstruction is predicated on the following idea: if a file A is created at time T,
then any files opened by the same user at times immediately before T are possible parents.

Another type of lineage relation, which we call acquaintance, happens when two files are
 opened nearby each other in the log by the same user. In this situation, there is no
 requirement that the openings be ordered in a certain way. The resulting graph is then not a directed graph.

The graph package we use, networkx, has the ability to add other attributes to the nodes and edges
of the directed graph. One can then, as needed, add other lineage/similarity measures to the edges.

Strength: Relatively fast, does not require access to files.
Weakness: Access log acts as proxy for lineage.


## Variable Matching

The final approach consists of a natural language processing analysis of variable descriptions in datasets.
The following diagram explains our process. The key point is that we take a dataset, pull-out its (user-created) variables,
and use gensim's FastText algorithm to compare those variables to variables in the metadata (again, not included here.)

For each one of the variables we've found that are similar to the user-created variables, we query the metadata to
take all files that contain those variables. We return these files as possible parents.

Variable matching is by far the slowest of all of the algorithms, but it is a good illustration of our approach
 to the lineage problems. Another advantage is that the algorithm doesn't have to access any sensitive data in the
 datasets themselves.
 
Strength: Only requires access to variables, not data itself. Theoretically interesting.
Weakness: Quite slow, as it must do a Word2Vec (and then WMDistance) query along millions of corpus variables.


## Citations

This work would not have been possible wihout the work of researchers inside and outside of the Census Bureau. We 
are especially indebted to those researchers who have worked hard to open-source their work.

Gensim:  
Software Framework for Topic Modelling with Large Corpora, Radim Rehurek and Petr Sojka, Proceedings of the LREC 2010 Workshop on New Challenges for NLP Frameworks.
      pp. 45--50. May 22,2010 http://is.muni.cz/publication/884893/en
      
Python:  
G. van Rossum, Python tutorial, Technical Report CS-R9526, Centrum voor Wiskunde en Informatica (CWI), Amsterdam, May 1995.

NumPy:  
Stéfan van der Walt, S. Chris Colbert and Gaël Varoquaux. The NumPy Array: A Structure for Efficient Numerical Computation, 
Computing in Science & Engineering, 13, 22-30 (2011), DOI:10.1109/MCSE.2011.37

Pandas:  
Wes McKinney. Data Structures for Statistical Computing in Python, Proceedings of the 9th Python in Science Conference, 51-56 (2010)

The concept of value-matching was largely informed by the following paper:  
Garfinkel, Simson L., and Michael McCarrin. "Hash-based carving: Searching media for complete files and file fragments with 
sector hashing and hashdb." Digital Investigation 14 (2015): S95-S105.
