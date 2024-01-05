#!/usr/bin/env python

import sys
import numpy as np

from pathlib import Path
file = Path(__file__).resolve()
parent = file.parent
sys.path.append(str(file.parent))
    
import subtree
import copy

DEBUG=False
precision = 10
    
    
def internalpathtrees(treefile, terminallist, numpathtrees):
    '''
    Generates a path between two trees using the geodesic
    '''
#    NOTE:  Results_subtrees has all subtrees of T1 and T2.
#           Each subtree has all information of it including
#           tips,edges,tipslength, edgelengths,
#           and supports(supports if two corresponding subtrees are disjoint)
    Results_subtrees, tip_dict , edge_dict = subtree.subtrees(treefile,terminallist)
    T1 = Results_subtrees[0]
    if DEBUG:
        print("T1",T1)
    T2 = Results_subtrees[1]
    disjoint_indices = Results_subtrees[2]
    if DEBUG:
        print("\nTree1 subtrees :\n",T1)
        print("\n\nTree2 subtrees :\n",T2)
        print("\n\ndisjoint_indices :\n",disjoint_indices)
    
    T1_tip_dict = tip_dict[0]
    T1_edge_dict = edge_dict[0]
    T2_tip_dict = tip_dict[1]
    T2_edge_dict = edge_dict[1]
    
    File = open(treefile, "r")
    file = File.readlines()
    treelist=np.array([s.replace('\n', '') for s in file])
    
    if DEBUG:
        print(f'\n++++++++++++++++++    Create PathTrees   ++++++++++++++++++\n')

    Lamda = np.linspace(0.0, 1.0, num=numpathtrees)
    Lamda = [round(elem,precision) for elem in Lamda ]
    thetreelist = []
    
    if DEBUG:
        print("Lamda", Lamda)
        print(f"\n\nnumber of subtrees in start tree = {len(T1)}")
        print(f"number of subtrees in end tree = {len(T2)}")
        print(f"\n\nsubtrees in start tree = {T1}")
        print(f"\nsubtrees in end tree = {T2}")
    
    for l, lamda in enumerate(Lamda[1:-1]):
        if DEBUG:
            print(f'\n=================  PathTree #{l}, lamda={lamda} ================\n')

        T1_path = copy.deepcopy(T1)
        for num in range(len(T1_path)):
            if num in (disjoint_indices[0]):
                if DEBUG:
                    print(f'\n~~~~~~~~~~~~~~~~ subtree #{num}    ,    lamda{lamda} ~~~~~~~~~~~~~~~~~~\n')
                    print(f'-------->  NOTE:  IS inside disjoint_indices \n ')
                    print(f"\n\nlen(T1[num]) = {len(T1[num])} \nlen(T2[num]) = {len(T2[num])} ")
                lambda_limits, epsilon = subtree.path_legs(num,  T1, T2, T1_edge_dict, T2_edge_dict)
                edited_subtree = subtree.pathtree_edges(num, T1, T2, T1_edge_dict, T2_edge_dict, lamda, lambda_limits, epsilon)
                T1_path[num] = edited_subtree
            else:
                if DEBUG:
                    print(f'\n~~~~~~~~~~~~ common subtree num #{num}    ,    lamda{lamda} ~~~~~~~~~~~~~~\n')
                    print(f'-------->  NOTE:  NOT inside disjoint_indices \n ')
                    print(f"\n----> len(T1[num]) = {len(T1[num])} \n----> len(T2[num]) = {len(T2[num])} ")

                T1_path[num][-2] = list( (1-lamda)*(np.array(T1[num][-2]))  + lamda*(np.array(T2[num][-2]) ) )   # length of tips(leaves) " (1-lambda)*e_T +lambda*e_T' "
                T1_path[num][-2] = [round(elem,precision) for elem in T1_path[num][-2] ]
                T1_path[num][-1] = list( (1-lamda)*(np.array(T1[num][-1]))  + lamda*(np.array(T2[num][-1]) ) )   #length of each common edge " (1-lambda)*e_T +lambda*e_T' "
                T1_path[num][-1] = [round(elem,precision) for elem in T1_path[num][-1] ]
        if DEBUG:
            print(f'\n~~~~~~~~~~~~~~~~~~ generated pathtree #{l} ~~~~~~~~~~~~~~~~~~~~~\n')
            print(f"\n----> Origional T1_path : \n{T1} ")
            print(f"\n----> Generated pathtree from T1 :\n{T1_path} ")
                
        sub_newicks , newick = subtree.Sub_Newicks(T1_path, disjoint_indices[0] )
        
        if DEBUG:
            print("\nsubtree newicks of path tree :\n",sub_newicks)
            print("\nnewick of pathtree :\n", newick)

        thetreelist.append(newick)
    return thetreelist
    

if __name__ == "__main__":

    if len(sys.argv)<2:
        print("pathtrees.py treefile terminalist")
    treefile = sys.argv[1]
    terminallist = sys.argv[2]
    numpathtrees=10
    mypathtrees='output_path'
    mypathtrees = internalpathtrees(treefile, terminallist, numpathtrees)
    print("Standalone test:\nlook at",mypathtrees)
