import numpy as np
import dendropy
from dendropy.calculate import treecompare
import time
import sys



RUN_PARALLEL = False

def run_wRF_pair1(t1,t2):  #unweighted
    t2.encode_bipartitions()
    d=treecompare.unweighted_robinson_foulds_distance(t1, t2, is_bipartitions_updated=True)
    return d

def run_wRF_pair2(t1,t2):    #weighted
    t2.encode_bipartitions()
    d=treecompare.weighted_robinson_foulds_distance(t1, t2, edge_weight_attr='length', is_bipartitions_updated=True)
    return d

def run_wRF_pair(a):
    t1,t2 = a
    t2.encode_bipartitions()
    d=treecompare.weighted_robinson_foulds_distance(t1, t2, edge_weight_attr='length', is_bipartitions_updated=True)
    return d

def process(tmptreelist):
    with ThreadPoolExecutor(max_workers=8) as executor:
        return  executor.map(run_wRF_pair, tmptreelist, timeout=60)

def RF_distances(n, filename_treelist, type="weighted"):
    tic1 = time.perf_counter()
    tns = dendropy.TaxonNamespace()
    distance_matrix=np.zeros((n,n))
    f=open(filename_treelist, 'r')
    tlst = dendropy.TreeList()
    trees=tlst.get(file=f,schema="newick",taxon_namespace=tns)
    f.close()
    toc1 = time.perf_counter()
    time1 = toc1 - tic1
#    print(f"Time of reading {n} trees= {time1}")
    
    tic2 = time.perf_counter()
    for i in range(1,n):
        trees[i].encode_bipartitions()
        t1 = trees[i]
        if RUN_PARALLEL:
            tmptreelist = [(trees[i],trees[j]) for j in range(i)]
            dlist = process(tmptreelist)
            distance_matrix[i][:i] = list(dlist)
        else:
            for j in range(i):
                if type == "weighted":
                    d = run_wRF_pair2(t1,trees[j])    #weighted
                else:
                    d = run_wRF_pair1(t1,trees[j])     #unweighted
                distance_matrix[i][j] = d
                distance_matrix[j][i] = d
    toc2 = time.perf_counter()
    time2 = toc2 - tic2
#    print(f"Time of distance matrix of {n}-trees using {type}-RF = {time2}")
    return distance_matrix
