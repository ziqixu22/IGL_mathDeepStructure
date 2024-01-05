import sys
import random
import math
import numpy as np
import itertools
import copy
import tree
import splittree
eps = 0.00001



def flatten(A):    #returns a flattened iterable containing all the elements of the input iterable (hear, it remove the biggest sublists brackets inside a nested list).
    rt = []
    for i in A:
        if isinstance(i,list): rt.extend(flatten(i))
        else: rt.append(i)
    return rt
    

def bifurcating(T_tips, T_edges, T_tipsLength, T_edgesLength):
    all_edges = T_edges.copy()
    all_edges.append(T_tips)
    sorted_edges = sorted(all_edges, key=len)
    new_edgelist = T_edges.copy()
    new_edgesLength = T_edgesLength.copy()
    
    for i, edge in enumerate(sorted_edges):
        remain = edge
        new_edge=[]
        for j in range(i-1,-1,-1):
            if set(sorted_edges[j]) <= set(remain):
                remain=sorted(list(set(remain)-set(sorted_edges[j])))
                new_edge.append(sorted_edges[j])
        new_edge.extend(remain)

        if len(new_edge)>2:
            for i in range(2, len(new_edge)):
                new = new_edge[1:i+1]
                new = flatten(new)
                new_edgelist.append(new)
                new_edgesLength.append(0)
    return([T_tips, new_edgelist, T_tipsLength, new_edgesLength])



def bifurcating_newick(treelist):      #multifurcating tree to bifurcating tree newick
    treelist_new=[]
    T = treelist
    for k in range(len(T)):
        p = tree.Node()
        p.name = 'root'
        mytree = tree.Tree()
        mytree.root=p
        mytree.myread(T[k],p)
        
        edges=[]
        edgelengths=[]
        edges, edgelengths =  mytree.get_edges(p, edges,edgelengths)
        edges = [ sorted(edge) for edge in edges ]
        
        tips=[]
        tiplengths=[]
        tips, tiplengths = mytree.getTips(p, tips, tiplengths)
        tipnames=[item.name for item in tips]
        dict_tips = dict(zip([str(i) for i in tipnames] , tiplengths))
        tipnames = sorted(tipnames)
        tiplengths = [dict_tips[str(a)] for a in tipnames]
        
        edges.append(sorted(tipnames))    #add root as an edge with length zero
        edgelengths.append(0.0)
        dict_edges = dict(zip([str(i) for i in edges] , edgelengths))
        edges_sorted = sorted(edges, key=len)[::-1]     #descending sort based on len of sublists
        edges_sorted = sorted(edges_sorted,key=lambda x: x[0] )
        edgeLengths_sorted = [dict_edges[str(a)] for a in edges_sorted]
        T1 = bifurcating(tipnames, edges_sorted[1:], tiplengths, edgeLengths_sorted[1:])
        
        newick_new = splittree.print_newick_string(T1[0], T1[1], T1[2], T1[3] )
        newick_new = newick_new+';'
        treelist_new. append(newick_new)
    return treelist_new



def bifur_to_mulfur_newick(treelist):    #bifurcating tree to multifurcating tree newick
    treelist_new=[]
    T = treelist
    for k in range(len(T)):
        p = tree.Node()
        p.name = 'root'
        mytree = tree.Tree()
        mytree.root=p
        mytree.myread(T[k],p)
        treelen = mytree.tree_len
        
        edges=[]
        edgelengths=[]
        edges, edgelengths =  mytree.get_edges(p, edges,edgelengths)
        edges = [ sorted(edge) for edge in edges ]
                
        tips=[]
        tiplengths=[]
        tips, tiplengths = mytree.getTips(p, tips, tiplengths)
        tipnames=[item.name for item in tips]
        
        dict_tips = dict(zip([str(i) for i in tipnames] , tiplengths))
        tipnames = sorted(tipnames)
        tiplengths = [dict_tips[str(a)] for a in tipnames]
        
        #bifurcate to multifurcate
        T_edg=[]
        T_edglen=[]
        for i, el in enumerate(edgelengths):
            if el>eps:
                T_edg.append(edges[i])
                T_edglen.append(edgelengths[i])
        newick_new = splittree.print_newick_string(tipnames, T_edg, tiplengths, T_edglen )
        newick_new = newick_new+';'
        treelist_new. append(newick_new)
    return treelist_new
    
    

if __name__ == "__main__":

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~Test1 : bifurcating ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    T_edges = [['a1', 'a2', 'a3', 'a4'], ['a1', 'a2']]
    T_edgesLength = [2,3]
    T_tips = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7']
    T_tipsLength = [1,1,1,1,1,1,1]
    result= bifurcating(T_tips, T_edges, T_tipsLength, T_edgesLength)
    print("Result :", result)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~Test2 : bifurcating_newick ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#    with open("paup_tree",'r') as myfile:
#        pauptree = myfile.readlines()
#    new_paup= bifurcating_newick(pauptree)
#    print("new_paup = ",  new_paup)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~Test3 : bifurcating_newick ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#    T=['((a1:1,a2:1,a3:1,a4:1):2,(a7:1,a8:1,(a9:1,a10:1):2):2,a5:1,a6:1,):0;']
#    new_T= bifurcating_newick(T)
#    print("\n\nT = ",  new_T)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~Test4 : bifur_to_mulfur_newick ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#    T=['(((a10:1.0,a9:1.0):2.0,(a7:1.0,a8:1.0):0):2.0,(a6:1.0,(a5:1.0,(a1:1.0,(a4:1.0,(a2:1.0,a3:1.0):0):0):2.0):0):0):0.0;']
#    new_T= bifur_to_mulfur_newick(T)
#    print("\n\nT = ",  new_T)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~Test5 : bifur_to_mulfur_newick ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#    File = open("our1", "r")
#    data = File.readlines()
#    T = [data[i].strip() for i in range(len(data))]
#    new_T= bifur_to_mulfur_newick(T)
#    print("\n\nT = ",  new_T)









    

