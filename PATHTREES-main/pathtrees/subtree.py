#!usr/bin/env python

import sys
from pathlib import Path
file = Path(__file__).resolve()
parent = file.parent
sys.path.append(str(file.parent))

#import random
#import math
import numpy as np
import tree
import itertools
import splittree
import copy
import bifurcating       

DEBUG=False
precision = 10


def subtrees(treelist, terminallist):
    File = open(treelist, "r")
    file = File.readlines()
    T=np.array([s.replace('\n', '') for s in file])

    Edges_list=[]
    EdgeLengths_list=[]

    Tip_list=[]
    TipLengths_list=[]

    for k in [0,1]:
        if DEBUG:
            print(f'\n================================= Tree {k+1} ===================================\n')
        p = tree.Node()
        p.name = 'root'
        mytree = tree.Tree()
        mytree.root=p
        mytree.myread(T[k],p)
        
        edges=[]
        edgelengths=[]
        Edges, Edgelengths =  mytree.get_edges(p, edges,edgelengths)
        Edges = [ sorted(edge) for edge in Edges ]

        tips=[]
        tipslength=[]
        tips, tipslength = mytree.getTips(p, tips, tipslength)
        tipnames=[tips[i].name for i in range(len(tips))]

        Edges.append(sorted(tipnames))    #add root as an edge with length zero
        Edgelengths.append(0.0)
        if DEBUG:
            print(f"\n\nTree{k+1} :", T[k])
            print("\n---->len(Edges) :", len(Edges))
            print("\n---->len(Edges length) :", len(Edgelengths))
            print("\n---->len(Tips) :", len(tipnames))
            print("\n---->len(Tips length) :", len(tipslength))
        Edges_list. append(Edges)
        EdgeLengths_list. append(Edgelengths)

        Tip_list. append(tipnames)
        TipLengths_list. append(tipslength)

    if DEBUG:
        print(f'\n=================== Commom & Uncommon edges with lengths ====================\n')

    Common_Edges = [edge for edge in Edges_list[0] if edge in Edges_list[1]]

    Common_indices = []
    Common_Edges_length = []
    Uncommon_Edges = []
    Uncommon_indices = []
    Uncommon_Edges_length = []
    for i in range(2):
        Common_indices.append([Edges_list[i].index(edge) for edge in Common_Edges])
        Common_Edges_length.append([EdgeLengths_list[i][l] for l in Common_indices[i]] )

        Uncommon_Edges.append([x for x in Edges_list[i] if x not in Common_Edges])
        Uncommon_indices.append([Edges_list[i].index(edge) for edge in Uncommon_Edges[i]])
        Uncommon_Edges_length.append([EdgeLengths_list[i][l] for l in Uncommon_indices[i]] )

    sub_list=[]
    Results=[]
    sub_dict=[]
    edge_dict=[]
    tip_dict=[]
    for t in range(2):    #two trees
        if DEBUG:
            print(f'\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  Tree{t+1}   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')

        edge_list = sorted(Edges_list[t], key=len).copy()    #all edges of tree
        common_list = sorted(Common_Edges, key=len)      #common edges of tree
        uncommon_list = ([sorted(uncommonedges, key=len) for uncommonedges in Uncommon_Edges])[t]      #uncommon edges of tree

        dict_edges = dict(zip([str(i) for i in Edges_list[t]] , EdgeLengths_list[t]))
        dict_tips = dict(zip([str(i) for i in Tip_list[t]]  , TipLengths_list[t]))
        dict_subs ={}

        edges_new = edge_list.copy()
        common_new = common_list.copy()
        uncommon_new = uncommon_list.copy()

        subT_edges = []      #disjoint subtrees
        subT_tips = []
        subT_edges_length = []
        subT_tips_length = []
        i =0
        while (len(common_new)>0):
            edge = common_new[0]

            if len(edge)==2:
                subtree = [edge]
                dict_tips.update( {'subT' + str(i) : dict_edges[str(edge)]} )
            elif len(edge)>2:
                subtree = [edge]
                dict_tips.update( {'subT' + str(i) : dict_edges[str(edge)]} )

                for j, item in enumerate(common_new):
                    if set(item) < set(edge):
                        subtree.append('subT'+ str(j))

                unwanted=[]
                for j, item in enumerate(uncommon_new):
                    if set(item) <= set(edge):
                        subtree.append(item)
                        unwanted.append(item)
                uncommon_new = [ele for ele in uncommon_new if ele not in unwanted]


            for j, item in enumerate(common_new):
                if set(edge) < set(item):
                    common_new[j]= ['subT'+ str(i)]
                    common_new[j].extend(list(set(item)-set(edge)))
                    dict_edges.update( {str(common_new[j]) : dict_edges[str(item)]} )

            for j, item in enumerate(uncommon_new):
                if set(edge) < set(item):
                    uncommon_new[j]= ['subT'+ str(i)]
                    uncommon_new[j].extend(list(set(item)-set(edge)))
                    dict_edges.update( {str(uncommon_new[j]) : dict_edges[str(item)]} )

            common_new.remove(edge)
            common_new = sorted(common_new, key=len)

            edges_new = sorted(common_new + uncommon_new, key=len)

            subT_edges.append(subtree)
            subT_tips.append(edge)

            i=i+1

        subT_edges_length=[]
        subT_tips_length =[]
        for i in range(len(subT_tips)):
            subT_tips_length.append([dict_tips[item] for item in subT_tips[i]])
            subT_edges_length.append([dict_edges[str(item)] for item in subT_edges[i]])

        if DEBUG:
            print("\n\n\n----> len(subtrees tip list) :\n", len(subT_tips))
            print("\n----> list of length of each subtree tips :\n",[len(item ) for item in subT_tips_length])
            print("\n----> len(subtrees edge list) :\n",len(subT_edges))
            print("\n----> list of length of each subtree edges :\n",[len(item ) for item in subT_edges_length])

        result_tree = [str(subT_tips), str(subT_edges), str(subT_tips_length), str(subT_edges_length)]

        s = [subT_tips, subT_edges, subT_tips_length, subT_edges_length]
        Results.append(s)
        
        sub_list.append(subT_edges)
        
        sub_dictionary = {}
        for i in range(len(subT_tips)):
            sub_dictionary['subT' + str(i)] = subT_tips[i]
        sub_dict.append(sub_dictionary)

        edge_dict.append(dict_edges)
        tip_dict.append(dict_tips)

    sub_list = [[sorted(sub_list[i][j], key=len)[::-1] for j in range(len(sub_list[i]))] for i in range(2) ]
    for s, subT in enumerate([sub_list[0], sub_list[1]]):
        for i in range(1,len(subT)):
            for j, item in enumerate(subT[i]):
                for k in range(len(subT)):
                    item = [subT[k][0] if x==str('subT' + str(k)) else [x] for x in item]
                    item = list(itertools.chain.from_iterable(item))
                    subT[i][j]=item
    if DEBUG:
        print("\n\n\n----> len(sub_list[0]) :", len(sub_list[0]))
        print("----> len(sub_list[1]) :", len(sub_list[1]))
        print(f'\n\n******************************  Extract GTP Informations ********************************\n')

    Combinatorial=[]
    start_tree=[]
    end_tree=[]

    mylines = []
    with open (terminallist, 'rt') as myfile:
        for myline in myfile:
            mylines.append(myline.rstrip('\n'))

    index=[]
    for i,line in enumerate(mylines):
        if line.lower().find("Starting tree edges:".lower()) != -1:       # If a match is found
            index.append(i+2)
        if line.lower().find("Target tree edges:".lower()) != -1:
            index.append(i+2)
        if line.lower().find("Leaf contribution squared".lower()) != -1:
            index.append(i)

    for line in mylines[index[0]:index[1]-3]:
        start_tree.append(line)
    start_tree=[s.split('\t\t') for s in start_tree]
    start_tree=[[s[0] , s[-1].split(',')] for s in start_tree]

    for line in mylines[index[1]:index[2]-1]:
        end_tree.append(line)
    end_tree=[s.split('\t\t') for s in end_tree]
    end_tree=[[s[0] , s[-1].split(',')] for s in end_tree]

    for line in mylines[index[2]:]:
        if line.lower().find("Combinatorial type:".lower()) != -1:
            Combinatorial.append(line)

    Combinatorial=[(support.split(': '))[1].split(';')[:-1] for support in  Combinatorial]

    supports=[]
    for s in Combinatorial:
        A=[];B=[]
        for item in s:
            item = item.rstrip().split('/')
            A.append(item[0][1:-1].split(','))
            B.append(item[1][1:-1].split(','))
        supports.append([A,B])

    dict_GTP1 = dict(zip([str(i[0]) for i in start_tree] , [str(i[1]) for i in start_tree]))
    dict_GTP2 = dict(zip([str(i[0]) for i in end_tree] , [str(i[1]) for i in end_tree]))

    A_list = []
    B_list = []

    for support in supports:
        A = support[0]
        B = support[1]
        A = [[dict_GTP1[a[i]] for i in range(len(a))] for a in A]
        B = [[dict_GTP2[b[i]] for i in range(len(b))] for b in B]
        item_A = [[eval(s.rstrip()) for s in l] for l in A]       # "eval" to convert string representation of list to a list
        item_B = [[eval(s.rstrip()) for s in l] for l in B]
        A_list.append(item_A)
        B_list.append(item_B)

    support_list = [A_list , B_list]

    flatten = itertools.chain.from_iterable
    flatten_list = []
    for i in range(2):
        flat=[]
        for item in support_list[i]:
            result = list(flatten(list(flatten(item))))
            result = list(set(result))      #to remove duplicates in the lists
            flat.append(result)
        flatten_list.append(flat)
    flatten_A = flatten_list[0]
    flatten_B = flatten_list[1]

    if DEBUG:
        print( "\n----> number of supports -----> len(flatten_A) :", len(flatten_A), "\n----> number of supports -----> len(flatten_B) :", len(flatten_B))
        print(f'\n\n********************  Add GTP Supports to "Results/subtrees file"  *********************\n')

    T1 = Results[0]
    T1 = [[item[i] for item in T1] for i in range(len(T1[0]))]
    T2 = Results[1]
    T2 = [[item[i] for item in T2] for i in range(len(T2[0]))]

    T1_editted = sub_list[0]
    T2_editted = sub_list[1]

    if DEBUG:
        print(f"\n\n~~~~~~~~~~~~~~~~~~ T1 , T2 disjoint ~~~~~~~~~~~~~~~~~~~~~~")
        
    disjoint_indices_T1=[]
    for i in range(len(flatten_A)):
        GTP_sub_tips = set(flatten_A[i]+flatten_B[i])
        for j, sub in enumerate(T1_editted):
            if ( set(GTP_sub_tips) <= set(sub[0])  ):
                disjoint_indices_T1.append(j)
                T1[j].append(A_list[i])
                T2[j].append(B_list[i])
                break
            
    disjoint_indices_T2 = disjoint_indices_T1
    if DEBUG:
        print("-----> len(disjoint_indices_T1) = ",len(disjoint_indices_T1))
        print("-----> disjoint_indices_T1 = ",disjoint_indices_T1, "\n\n")
        print(f'\n\n***************  Test : GTP Support list based on subs"  ****************\n')

    Tree1 = Results[0]
    Tree1 = [[item[i] for item in Tree1] for i in range(len(Tree1[0]))]
    Tree2 = Results[1]
    Tree2 = [[item[i] for item in Tree2] for i in range(len(Tree2[0]))]

    A_list_editted = A_list.copy()
    M1 = list(sub_dict[0].keys())
    for c in A_list_editted:
        for b in c:
            for m in M1:
                for i, a in enumerate(b):
                    if set(sub_dict[0][m]) <= set(a):
                        b[i]=[m]
                        b[i].extend(list(set(a)-set(sub_dict[0][m])))
    B_list_editted = B_list.copy()
    M2 = list(sub_dict[1].keys())
    for c in B_list_editted:
        for b in c:
            for m in M2:
                for i, a in enumerate(b):
                    if set(sub_dict[0][m]) <= set(a):
                        b[i]=[m]
                        b[i].extend(list(set(a)-set(sub_dict[1][m])))
    if DEBUG:
        print("\n\n\n----> len(A_list) :\n", len(A_list))
        print("\n----> len(B_list) :\n", len(B_list))

    Results_subtrees =[T1, T2, [sorted(disjoint_indices_T1) , sorted(disjoint_indices_T2)]]

    return(Results_subtrees, tip_dict, edge_dict)



def Sub_Newicks(T, disjoint_indices):
    sub_newicks_list  = []
    if DEBUG:
        print(f"\n\n-----> len(T1_path) : {len(T)}")
    for i in range(len(T)):       # BIFURCATING
        T1 = bifurcating.bifurcating(T[i][0], T[i][1][1:], T[i][2], T[i][3][1:])
        if i in disjoint_indices:
            newick_sub = splittree.print_newick_string(T1[0], T1[1], T1[2], T1[3] )
            if DEBUG:
                print(f"\nsubT{i} --> disjoint :", newick_sub)
        if i not in disjoint_indices:
            newick_sub = splittree.print_newick_string(T1[0], T1[1], T1[2], T1[3] )
            if DEBUG:
                print(f"\nsubT{i} --> NOT disjoint :", newick_sub)
        sub_newicks_list.append(newick_sub)
    if DEBUG:
        print("sub_newicks list :", sub_newicks_list)
    newick = sub_newicks_list[-1]
    for i in range(len(sub_newicks_list)-2, -1,-1):
        newick = newick.replace(str('subT' + str(i)), sub_newicks_list[i][:-4])
    newick = newick+';'
    return(sub_newicks_list, newick)



# subT1, subT2 are two disjoint trees ( we know all information of them including tips,edges,tipslength, edgelengths, supports)
def path_legs(num, T1, T2, T1_edge_dict, T2_edge_dict):      # lamda a number in [0,1]
    subT1 = T1[num]    # subT1, subT2 should be disjoint trees  (num: the numeber of disjoint pair trees that we want to study)
    subT2 = T2[num]
    A = subT1[-1]
    B = subT2[-1]
    if DEBUG:
        print(f"\n===> starting subtree :\n {subT1}")
        print(f"\n===> ending subtree :\n {subT2}")
    k= len(A)
    if DEBUG:
        print(f"\n\n===> k = {k}")
        print(f"\n===> number of legs (orthants) = k+1 = {k+1}")
    lambda_limits = [0]
    for i in range(k):
        A_i=[T1_edge_dict[str(e)] for e in A[i]]
        B_i=[T2_edge_dict[str(e)] for e in B[i]]
        lambda_limits.append( np.linalg.norm(A_i)/(np.linalg.norm(A_i) + np.linalg.norm(B_i)))
    lambda_limits.append(1)
    lambda_limits = [round(elem,precision) for elem in lambda_limits ]

    epsilon=[A]
    for i in range(1,k):
        epsilon.append(list(itertools.chain.from_iterable([B[0:i] , A[i:k]])))
    epsilon.append(B)
    if DEBUG:
        print(f"\n===> lambda_limits = {lambda_limits}")
    return(lambda_limits, epsilon)





def pathtree_edges(num, T1, T2, T1_edge_dict, T2_edge_dict,lamda, lambda_limits, epsilon):
    subT1 = T1[num]
    subT2 = T2[num]
    A = subT1[-1]
    B = subT2[-1]
    A_flatten = list(itertools.chain.from_iterable(A))  # "itertools.chain.from_iterable" returns a flattened iterable containing all the elements of the input iterable (hear, it remove the biggest sublists brackets inside a nested list).
    B_flatten = list(itertools.chain.from_iterable(B))
    if DEBUG:
        print(f"\n\n===> A = {A}")
        print(f"\n===> A_flatten = {A_flatten}")
        print(f"\n===> B = {B}")
        print(f"\n===> B_flatten = {B_flatten}")
    
    k= len(A)
    flatten_epsilon = [ list(itertools.chain.from_iterable(item)) for item in epsilon]
    if DEBUG:
        print(f"\n\n\n===> epsilon :\n {epsilon}")
        print(f"\n===> flatten_epsilon :\n {flatten_epsilon}")

    i = [j for j in range(k+1) if lambda_limits[j]<= lamda <=lambda_limits[j+1]][0]    # i in [0:k]----> number of leg that lamda is there
    if DEBUG:
        print(f"\n\n\n===> for lambda {lamda}: \n i={i} \nsubtree is on the leg(i={i})= {flatten_epsilon[i]}  \n(NOTE: leg index starting from zero)\n\n")
    
    EdgeLength_i=[]
    if i<(k):
        if lamda < lambda_limits[i+1]:
            if DEBUG:
                print("\n***********  NOTE:  Not on the border  ***********" )
            for  edge in flatten_epsilon[i]:
                if DEBUG:
                    print(f"\n\nFor edge in flatten_epsilon[i={i}] -----> edge = {edge}")
                if edge in A_flatten:
                    j = next(i for i, v in enumerate(A) if edge in v)
                    if DEBUG:
                        print(f"edge in  A_j = A_{j}")
                        print("A[j] = ", A[j])
                        print("B[j] = ", B[j])
                    Aj_norm = np.linalg.norm([T1_edge_dict[str(e)] for e in A[j]])
                    if DEBUG:
                        print("Aj_norm = ", Aj_norm)
                    Bj_norm = np.linalg.norm([T2_edge_dict[str(e)] for e in B[j]])
                    if DEBUG:
                        print("Bj_norm = ", Bj_norm)
                    EdgeLength_i.append( (((1-lamda)*Aj_norm-lamda*Bj_norm)/Aj_norm)*(T1_edge_dict[str(edge)])  )
                    if DEBUG:
                        print(f"new edge_length = {(((1-lamda)*Aj_norm-lamda*Bj_norm)/Aj_norm)*(T1_edge_dict[str(edge)])}")
                if edge in B_flatten:
                    j = next(i for i, v in enumerate(B) if edge in v)    #"j" : the index of B_j of B that edge is inside that
                    if DEBUG:
                        print(f"edge in  B_j = B_{j}")
                        print("A[j] = ", A[j])
                        print("B[j] = ", B[j])
                    Aj_norm = np.linalg.norm([T1_edge_dict[str(e)] for e in A[j]])
                    if DEBUG:                         
                        print("Aj_norm = ", Aj_norm)
                    Bj_norm = np.linalg.norm([T2_edge_dict[str(e)] for e in B[j]])
                    if DEBUG:
                        print("Bj_norm = ", Bj_norm)
                    EdgeLength_i.append( ((lamda*Bj_norm-(1-lamda)*Aj_norm)/Bj_norm)*(T2_edge_dict[str(edge)])  )
                    if DEBUG:
                        print(f"new edge_length = {((lamda*Bj_norm-(1-lamda)*Aj_norm)/Bj_norm)*(T2_edge_dict[str(edge)])}")

        if lamda == lambda_limits[i+1]:
            if DEBUG:
                print("\n*********** NOTE:  On the border  ***********")
            border_ZeroEdges = [e for e in flatten_epsilon[i] if e not in flatten_epsilon[i+1]]
            if DEBUG:
                print(f"\nborder_ZeroEdges = {border_ZeroEdges}")
            for  edge in flatten_epsilon[i]:
                if DEBUG:
                    print(f"\n\nFor edge in flatten_epsilon[i={i}] -----> edge = {edge}")
                if edge in border_ZeroEdges:
                    EdgeLength_i.append(0.0)
                else:
                    if edge in A_flatten:
                        j = next(i for i, v in enumerate(A) if edge in v)
                        if DEBUG:
                            print(f"edge in  A_j = A_{j}")
                            print("A[j] = ", A[j])
                            print("B[j] = ", B[j])
                        Aj_norm = np.linalg.norm([T1_edge_dict[str(e)] for e in A[j]])
                        if DEBUG:
                            print("Aj_norm = ", Aj_norm)
                        Bj_norm = np.linalg.norm([T2_edge_dict[str(e)] for e in B[j]])
                        if DEBUG:
                            print("Bj_norm = ", Bj_norm)
                        EdgeLength_i.append( (((1-lamda)*Aj_norm-lamda*Bj_norm)/Aj_norm)*(T1_edge_dict[str(edge)])  )
                        if DEBUG:
                            print(f"new edge_length = {(((1-lamda)*Aj_norm-lamda*Bj_norm)/Aj_norm)*(T1_edge_dict[str(edge)])}")
                    if edge in B_flatten:
                        j = next(i for i, v in enumerate(B) if edge in v)    #"j" : the index of B_j of B that edge is inside that
                        if DEBUG:
                            print(f"edge in  B_j = B_{j}")
                            print("A[j] = ", A[j])
                            print("B[j] = ", B[j])
                        Aj_norm = np.linalg.norm([T1_edge_dict[str(e)] for e in A[j]])
                        if DEBUG:
                            print("Aj_norm = ", Aj_norm)
                        Bj_norm = np.linalg.norm([T2_edge_dict[str(e)] for e in B[j]])
                        if DEBUG:
                            print("Bj_norm = ", Bj_norm)
                        EdgeLength_i.append( ((lamda*Bj_norm-(1-lamda)*Aj_norm)/Bj_norm)*(T2_edge_dict[str(edge)])  )
                        if DEBUG:
                            print(f"new edge_length = {((lamda*Bj_norm-(1-lamda)*Aj_norm)/Bj_norm)*(T2_edge_dict[str(edge)])}")
    if i==k:
        if DEBUG:
            print("\n***********  NOTE:  On the last leg  ***********" )
        for  edge in flatten_epsilon[i]:
            if DEBUG:
                print(f"\n\nFor edge in flatten_epsilon[i={i}] -----> edge = {edge}")
            j = next(i for i, v in enumerate(B) if edge in v)    #"j" : the index of B_j of B that edge is inside that
            if DEBUG:
                print(f"edge in  B_j = B_{j}")
                print("A[j] = ", A[j])
                print("B[j] = ", B[j])
            Aj_norm = np.linalg.norm([T1_edge_dict[str(e)] for e in A[j]])
            if DEBUG:
                print("Aj_norm = ", Aj_norm)
            Bj_norm = np.linalg.norm([T2_edge_dict[str(e)] for e in B[j]])
            if DEBUG:
                print("Bj_norm = ", Bj_norm)
            EdgeLength_i.append( ((lamda*Bj_norm-(1-lamda)*Aj_norm)/Bj_norm)*(T2_edge_dict[str(edge)])  )
            if DEBUG:
                print(f"new edge_length = {((lamda*Bj_norm-(1-lamda)*Aj_norm)/Bj_norm)*(T2_edge_dict[str(edge)])}")


    EdgeLength_i = [round(elem,precision) for elem in EdgeLength_i ]
    edited_subtree = copy.deepcopy(subT1)     #"deepcopy" to make sure the changes in copy does not effect origional list
    edited_subtree[1][1:] = flatten_epsilon[i]
    
    edited_subtree[3][0] = round((1-lamda)*(subT1[3][0])  + lamda*(subT2[3][0])  , precision)  #length of each common edge " (1-lambda)*e_T +lambda*e_T' "
    edited_subtree[3][1:] = EdgeLength_i         # length of edges(disjoint edges) in the new orthant

    edited_subtree[2] = list( (1-lamda)*(np.array(subT1[2]))  + lamda*(np.array(subT2[2]) ) )  # length of tips(leaves) " (1-lambda)*e_T +lambda*e_T' "
    edited_subtree[2] = [round(elem,precision) for elem in edited_subtree[2] ]
    if DEBUG:
        print("\n\nstarting subtree :\n", subT1)
        print("ending subtree :\n", subT2)
        print("\n\nEditted starting subtree :\n", edited_subtree)

    return(edited_subtree)     #The subtree number "num" in T1 should be replaced with this "edited_subtree" for the given "lamba".





if __name__ == "__main__":
    print("Standalone test")
    if len(sys.argv)<2:
        printf("needs two arguments: treelist and terminallist")
        sys.exit()
    treelist = sys.argv[1]
    terminallist = sys.argv[2]
    Results = subtrees(treelist,terminallist)
    print(Results)
    



