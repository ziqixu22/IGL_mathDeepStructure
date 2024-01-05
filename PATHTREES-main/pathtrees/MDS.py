import numpy as np
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.lines as mlines

from scipy.interpolate import griddata
from scipy.interpolate import Rbf
from scipy.spatial import ConvexHull

import statsmodels.api as sm
import scipy.stats

import warnings
warnings.filterwarnings("ignore")

mpl.rcParams['agg.path.chunksize'] = 100000

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Reading Files~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def read_from_files(likelihoodfile,treefile,pathtreefile,starttreefile):
    like = open(likelihoodfile, "rb")
    like = like.readlines()
    Likelihood = np.loadtxt(like)
    print("length of Likelihood= ", len(Likelihood))

    tree = open(treefile, "r") 
    treelist = tree.readlines()
    print("length of treelist : ", len(treelist))

    Path = open(pathtreefile, "rb")
    pathlist = Path.readlines()
    N= len(pathlist)
    print("N= ", len(pathlist))
    
    start = open(starttreefile, "r")
    StartTrees = start.readlines()
    print("length of StartTrees : ", len(StartTrees))
    
    return Likelihood,treelist,pathlist,StartTrees
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def read_GTP_distances(n,GTPOUTPUT):
    data=[]
    File = open(GTPOUTPUT, "r")
    file = File.readlines()
    for line in file:
        temp=line.rstrip().split('\t')
        data.append(temp[-1])
    data = [x for x in data if x != '']
    ind = np.triu_indices(n,k=1)
    M = np.zeros((n,n))
    M[ind]=data
    for i in range(len(M)):
        for j in range(len(M)):
            M[j][i]=M[i][j]
    distances=M
    return distances

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~MDS Algorithm~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def MDS(M,d):
    n=len(M)
    H=np.eye(n)-(1/n)*(np.ones((n,n)))
    K=-(1/2)*H.dot(M**2).dot(H)
    eval, evec = np.linalg.eig(K)
    idx = np.argsort(eval)[::-1] 
    eval = eval[idx][:d]
    evec = evec[:,idx][:, :d]
    X = evec.dot(np.diag(np.sqrt(eval)))
    X=X.real     #to get rid of imaginary part
    return X

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~likelihood index~~~~~~~~~~~~~~~~~~~~~~~
def best_likelihoods(Likelihood, n=100000000):   #default=100000000
    # n is NUMBESTTREES from options
    sort_index= sorted(range(len(Likelihood)), key=lambda k: Likelihood[k])
    if n >= len(sort_index):
        idx=sort_index
    else:
        idx=sort_index[-n:]
    return idx

def bestNstep_likelihoods(Likelihood, n, step):
    lenL = len(Likelihood)
    if n > lenL:
        n=lenL
    sort_index= sorted(range(len(Likelihood)), key=lambda k: Likelihood[k])
    idx=sort_index[-n::step]    # last m best, every step  ---> n=0 & step=1 : all trees
    Like_Big = [Likelihood[i] for i in idx]
    return idx
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MDS Validation ~~~~~~~~~~~~~~~~~~~~~~~
def MDS_validation(M,X,k, it):  # M: real distance matrix & X: MDS coordinates
    n=len(M)
    D = np.zeros((n,n))     # D: MDS distance matrix
    for i in range(n):
        for j in range(n):
            if i<j:
                D[i,j] = np.linalg.norm(np.abs(X[i]-X[j]))
    D=D+D.T
    x= M.flatten()
    y=D.flatten()

    x,y=zip( *sorted( zip(x, y) ) )
    x1=x[0::k]   # choose every kth of data
    y1=y[0::k]
    
    Pearson = scipy.stats.pearsonr(x, y)[0]    # Pearson's r
    Spearman = scipy.stats.spearmanr(x, y)[0]   # Spearman's rho
    Kendall = scipy.stats.kendalltau(x, y)[0]  # Kendall's tau
    print(f'\n>>> MDS validation ...')
    print(f"    Pearson's r = {Pearson} \n    Spearman's rho = {Spearman} \n    Kendall's tau = {Kendall}")
    MDS_validation_values = [Pearson,Spearman,Kendall]

    lowess = sm.nonparametric.lowess
    Z = lowess(y1,x1)

    fig = plt.figure(figsize=(7,5))
    plt.scatter(x,y, marker='.',s=0.07, color='orange')
    plt.plot(Z[:,0],Z[:,1],'.', markersize=0.05, color='teal')
    plt.xlabel('Distance in treespace')
    plt.ylabel('Distance on MDS')
#    plt.savefig('ShepardDiagram_iter{}.pdf'.format(it), format='pdf')
    plt.savefig('ShepardDiagram_iter{}.png'.format(it), format='png')
    plt.close(fig)
#    plt.show()
    return D, MDS_validation_values
    


def plot_MDS(plotfile, N, n, M,Likelihood, bestlike, treelist, pathlist):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~2D MDS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    X= MDS(M,2)
    fig = plt.figure(figsize=(12,5))
    axes=[None,None]

    fig.set_facecolor('white')
    plt.rcParams['grid.color'] = "white" # change color
    plt.rcParams['grid.linewidth'] = 1   # change linwidth
    plt.rcParams['grid.linestyle'] = '-'

    axes[0] = fig.add_subplot(1,2,1)
    s = axes[0].scatter(X[:-N,0], X[:-N,1],marker='^', c=Likelihood[:-N] , cmap='viridis', s=25)  #Not PathTrees
    axes[0].scatter(X[-N:,0], X[-N:,1], c=Likelihood[-N:] , cmap='viridis', s=6)  #PathTrees
    cbar = fig.colorbar(s)
    cbar.set_label('Log Likelihood', rotation=270 , labelpad=15)
    axes[0].set_xlabel('Coordinate 1')
    axes[0].set_ylabel('Coordinate 2')

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~3D MDS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    axes[1] = fig.add_subplot(1,2,2,projection='3d')
    X1= MDS(M,3)
    axes[1].scatter(np.real(X1[:-N,0]),np.real(X1[:-N,1]),np.real(X1[:-N,2]), marker='^',c=Likelihood[:-N] , alpha=0.8, cmap='viridis', s=25)#Not PathTrees
    axes[1].scatter(np.real(X1[-N:,0]),np.real(X1[-N:,1]),np.real(X1[-N:,2]),c=Likelihood[-N:] , alpha=0.8, cmap='viridis', s=6)#PathTrees
    print("Best Trees:\n")
    idx = list(zip(*bestlike))[0]
    for i in idx:
        axes[1].scatter(np.real(X1[i,0]),np.real(X1[i,1]),np.real(X1[i,2]),c='r',s=10)
        axes[1].text(np.real(X1[i,0]),np.real(X1[i,1]),np.real(X1[i,2]),i,size=7)
        print("tree #",i,"\n",pathlist[i])
    axes[1].set_xlabel('Coordinate 1')
    axes[1].set_ylabel('Coordinate 2')
    axes[1].set_zlabel('Coordinate 3')
    plt.tight_layout()
    plt.savefig(plotfile)
    #plt.show()

#=================== Interpolation method2: Griddata (Contour & Surface) ====================

def interpolate_grid(it, filename, M, Likelihood, bestlike, Treelist, StartTrees,  optimized_BestTrees, Topologies,NUMPATHTREES, dataset_option_list,dataset_option, validation_mds, hull_indices=None ):

    paup_tree = dataset_option_list[0]
    paup_MAP = dataset_option_list[1]
    paup_RXML = dataset_option_list[2]
    user_trees = dataset_option_list[3]
    
    meth= 'cubic'
#    meth= 'linear'
#    meth= 'nearest'
    n = len(Treelist)
    N = len(StartTrees)
    n_path = NUMPATHTREES-2
    if paup_tree:
        opt = len(optimized_BestTrees)+1    #just PAUP
    elif paup_MAP:
        opt = len(optimized_BestTrees)+2    #both PAUP&MAP
    elif paup_RXML:
        opt = len(optimized_BestTrees)+2    #both PAUP&RXML
    elif user_trees:
        with open(dataset_option,'r') as myfile:
            usertrees = myfile.readlines()
        opt = len(optimized_BestTrees)+len(usertrees)    #user_trees
    else:
        opt = len(optimized_BestTrees)
    
    all_path = n-opt-N
    num=200
    
    X= MDS(M,2)
    xx = np.linspace(np.min(X),np.max(X),num)
    yy =  np.linspace(np.min(X),np.max(X),num)
    XX, YY = np.meshgrid(xx,yy)          #We DONT know the likelihood values of these points
    values = griddata((np.real(X[:,0]), np.real(X[:,1])), Likelihood, (XX, YY), method=meth)

    N_topo = len(Topologies)
    colormap = plt.cm.RdPu
    Colors = [colormap(i) for i in np.linspace(0.1, 0.9,N_topo)]

    fig = plt.figure(figsize=(12,5))
    fig.set_facecolor('white')
    plt.rcParams['grid.color'] = "white" # change color
    plt.rcParams['grid.linewidth'] = 1   # change linwidth
    plt.rcParams['grid.linestyle'] = '-'

    ax1 = plt.subplot2grid((40,350), (14,0),rowspan=13, colspan=80)
    contour = ax1.contourf(XX,YY,values, 10, alpha=1, vmin=np.nanmin(values), vmax=np.nanmax(values), cmap='viridis')

    vmin, vmax = np.nanmin(values.flatten()), np.nanmax(values.flatten())
    
    ax1.scatter(X[:N,0], X[:N,1], alpha=1, marker='^',facecolors='none', vmin = vmin, vmax = vmax, cmap='viridis',  edgecolors="black", linewidths=0.7, s=60)     #starttrees

    points = ax1.scatter(X[N:,0], X[N:,1], c=Likelihood[N:] , alpha=1,  marker='o', cmap='viridis',  edgecolors="black", linewidths=0.3, s=5)  #PathTrees
    if paup_tree:
        ax1.scatter(X[-1,0],X[-1,1], marker='o',facecolors='white', edgecolors="black", linewidths=0.5, s=140)      #PAUP
        ax1.scatter(X[-2,0],X[-2,1], marker='D', facecolors='red', edgecolors="black", linewidths=0.5, s=220)     #best optimized
        opt_points = ax1.scatter(X[-opt:-1,0],X[-opt:-1,1], marker='o',c=Colors, edgecolors="black", linewidths=0.5, s=40)      #optimized trees
    
    elif paup_MAP:
        ax1.scatter(X[-3,0],X[-3,1], marker='D', facecolors='red', edgecolors="black", linewidths=0.5, s=220)     #best optimized
        ax1.scatter(X[-2,0],X[-2,1], marker='o',facecolors='white', edgecolors="black", linewidths=0.5, s=140)      #PAUP
        ax1.scatter(X[-1,0],X[-1,1], marker='o', facecolors='black', edgecolors="black", linewidths=0.5, s=140)     #MAP
        opt_points = ax1.scatter(X[-opt:-2,0],X[-opt:-2,1], marker='o',c=Colors, edgecolors="black", linewidths=0.5, s=40)      #optimized trees
        
    elif paup_RXML:
        ax1.scatter(X[-3,0],X[-3,1], marker='D', facecolors='red', edgecolors="black", linewidths=0.5, s=220)     #best optimized
        ax1.scatter(X[-2,0],X[-2,1], marker='o',facecolors='white', edgecolors="black", linewidths=0.5, s=140)      #PAUP
        ax1.scatter(X[-1,0],X[-1,1], marker='o', facecolors='black', edgecolors="black", linewidths=0.5, s=140)     #RXML_bropt
        opt_points = ax1.scatter(X[-opt:-2,0],X[-opt:-2,1], marker='o',c=Colors, edgecolors="black", linewidths=0.5, s=40)      #optimized trees
        
    elif user_trees:
        nn= len(usertrees)
        ax1.scatter(X[-(nn+1),0],X[-(nn+1),1], marker='D', facecolors='red', edgecolors="black", linewidths=0.5, s=220)     #best optimized
        for i in range(1, nn+1):
            ax1.scatter(X[-i,0],X[-i,1], marker='o',facecolors='white', edgecolors="black", linewidths=0.5, s=140)      #usertrees
        opt_points = ax1.scatter(X[-opt:-nn,0],X[-opt:-nn,1], marker='o',c=Colors, edgecolors="black", linewidths=0.5, s=40)      #optimized trees
    else:
        ax1.scatter(X[-1,0],X[-1,1], marker='D', facecolors='red', edgecolors="black", linewidths=0.5, s=200)     #best optimized
        opt_points = ax1.scatter(X[-opt:,0],X[-opt:,1], marker='o',c=Colors, edgecolors="black", linewidths=0.5, s=40)      #optimized trees
        
    for j, (top, c) in enumerate(zip(Topologies, Colors)):     #topologies trees
        ax1.scatter(X[top,0], X[top,1], marker='o', color=c, edgecolors="black", linewidths=0.15,s=8)
    
  
    min_xvalue, max_xvalue = np.min(X[:,0]),np.max(X[:,0])       #To hadle the distance between boundary and contour
    min_yvalue, max_yvalue = np.min(X[:,1]),np.max(X[:,1])
    x_dist =(max_xvalue-min_xvalue)/30
    y_dist =(max_yvalue-min_yvalue)/30
    
    ax1.grid(linestyle=':', linewidth='0.2', color='white', alpha=0.6)
    ax1.set_xlim(min_xvalue-x_dist, max_xvalue+x_dist)
    ax1.set_ylim(min_yvalue-y_dist, max_yvalue+y_dist)
    
    ax1.set_xlabel('Coordinate 1', labelpad=7, fontsize=8)
    ax1.set_ylabel('Coordinate 2', labelpad=3, fontsize=8)
    ax1.tick_params(labelsize=4.5)

    
    ax1.ticklabel_format(useOffset=False, style='plain')
    fig.autofmt_xdate()    #to fix the issue of overlapping x-axis labels
    
    
    if n_path==1:
        smallgreen_circle = mlines.Line2D([], [], color='limegreen', marker='o', linestyle='None',mec="black", mew=0.3, markersize=3, label="{} pathtrees + {} starting trees:\n{}-tree between each pair".format(all_path,N, n_path))
    else:
        smallgreen_circle = mlines.Line2D([], [], color='limegreen', marker='o', linestyle='None',mec="black", mew=0.3, markersize=3, label="{} pathtrees + {} starting trees:\n{}-trees between each pair".format(all_path,N, n_path))
    green_triangle = mlines.Line2D([], [], color='none', marker='^', linestyle='None',mec="black", mew=0.7, markersize=8, label="{} starting trees".format(len(StartTrees)))
    smallpink_circle = mlines.Line2D([], [], color='hotpink', marker='o', linestyle='None', mec="black", mew=0.25, markersize=3, label="{} trees for topology analysis ".format(len(bestlike)))
    bigpink_circle = mlines.Line2D([], [], color='hotpink', marker='o', linestyle='None',mec="black", mew=0.4, markersize=6, label="{} optimized trees  from different \ntopologies".format(N_topo))
    
    
    if paup_tree:
        bigred_square = mlines.Line2D([], [], color='red', marker='D', linestyle='None',mec="black", mew=0.7, markersize=11, label="PATHTREES best optimized tree \nLogLike {:4.4f}".format(Likelihood[-2]))
        bigwhite_circle = mlines.Line2D([], [],  marker='o', linestyle='None',mec="black",mfc='white', mew=0.9, markersize=11, label="PAUP best optimized tree \nLogLike {:4.4f}".format(Likelihood[-1]))
        plt.legend(bbox_to_anchor=(1.05, 0.95),  fontsize='x-small', labelspacing = 1.5, frameon=False, scatterpoints=1, handles=[ smallgreen_circle, green_triangle, smallpink_circle, bigpink_circle,   bigred_square, bigwhite_circle])
    
    elif paup_MAP:
        bigred_square = mlines.Line2D([], [], color='red', marker='D', linestyle='None',mec="black", mew=0.7, markersize=11, label="PATHTREES best optimized tree \nLogLike {:4.4f}".format(Likelihood[-3]))
        bigwhite_circle = mlines.Line2D([], [],  marker='o', linestyle='None',mec="black",mfc='white', mew=0.9, markersize=11, label="PAUP best optimized tree \nLogLike {:4.4f}".format(Likelihood[-2]))
        bigblack_circle = mlines.Line2D([], [],  marker='o', linestyle='None',mec="black",mfc='black', mew=0.9, markersize=11, label="MAP best optimized tree \nLogLike {:4.4f}".format(Likelihood[-1]))
        plt.legend(bbox_to_anchor=(1.05, 0.95),  fontsize='x-small', labelspacing = 1.5, frameon=False, scatterpoints=1, handles=[ smallgreen_circle, green_triangle, smallpink_circle, bigpink_circle,   bigred_square, bigwhite_circle, bigblack_circle])
        
        
    elif paup_RXML:
        bigred_square = mlines.Line2D([], [], color='red', marker='D', linestyle='None',mec="black", mew=0.7, markersize=11, label="PATHTREES best optimized tree \nLogLike {:4.4f}".format(Likelihood[-3]))
        bigwhite_circle = mlines.Line2D([], [],  marker='o', linestyle='None',mec="black",mfc='white', mew=0.9, markersize=11, label="PAUP best optimized tree \nLogLike {:4.4f}".format(Likelihood[-2]))
        bigblack_circle = mlines.Line2D([], [],  marker='o', linestyle='None',mec="black",mfc='black', mew=0.9, markersize=11, label="RAxML best optimized tree \nLogLike {:4.4f}".format(Likelihood[-1]))
        plt.legend(bbox_to_anchor=(1.05, 0.95),  fontsize='x-small', labelspacing = 1.5, frameon=False, scatterpoints=1, handles=[ smallgreen_circle, green_triangle, smallpink_circle, bigpink_circle,   bigred_square, bigwhite_circle, bigblack_circle])
        
    elif user_trees:
        nn= len(usertrees)
        bigred_square = mlines.Line2D([], [], color='red', marker='D', linestyle='None',mec="black", mew=0.7, markersize=11, label="PATHTREES best optimized tree \nLogLike {:4.4f}".format(Likelihood[-(nn+1)]))
        bigwhite_circle = mlines.Line2D([], [],  marker='o', linestyle='None',mec="black",mfc='white', mew=0.9, markersize=11, label="user trees")
        plt.legend(bbox_to_anchor=(1.05, 0.85),  fontsize='x-small', labelspacing = 1.5, frameon=False, scatterpoints=1, handles=[ smallgreen_circle, green_triangle, smallpink_circle, bigpink_circle,   bigred_square, bigwhite_circle])
    
    else:
        bigred_square = mlines.Line2D([], [], color='red', marker='D', linestyle='None',mec="black", mew=0.7, markersize=11, label="PATHTREES best optimized tree \nLogLike {:4.4f}".format(Likelihood[-1]))
        plt.legend(bbox_to_anchor=(1.05, 0.85),  fontsize='x-small', labelspacing = 1.5, frameon=False, scatterpoints=1, handles=[ smallgreen_circle, green_triangle, smallpink_circle, bigpink_circle,   bigred_square])
    plt.setp(ax1.spines.values(), color='gray')

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ax2 = plt.subplot2grid((40,350), (0,150), rowspan=50, colspan=200, projection='3d')
    ax2.plot_wireframe(XX, YY, values,  rstride=2, cstride=2, linewidth=1,alpha=0.3, cmap='viridis')
    surf = ax2.plot_surface(XX, YY, values,  rstride=1, cstride=1, edgecolor='none',alpha=0.5, cmap='viridis', vmin=np.nanmin(values), vmax=np.nanmax(values))

    min_value, max_value = ax2.get_zlim()       #To hadle the distance between surface and contour

    cset = ax2.contourf(XX, YY, values, vmin=np.nanmin(values), vmax=np.nanmax(values), zdir='z', offset=min_value, cmap='viridis', alpha=0.3)

    ax2.scatter3D(X[:N,0], X[:N,1], Likelihood[:N], marker='^',facecolors='none',  vmin = vmin, vmax = vmax, edgecolors="black", linewidths=0.5, s=50)    #starttrees
    
    if paup_tree:
        ax2.scatter3D(X[-1,0],X[-1,1],Likelihood[-1], marker='o',facecolors='white', edgecolors="black", linewidths=0.5, s=140)      #PAUP
        ax2.scatter3D(X[-2,0],X[-2,1],Likelihood[-2], marker='D',facecolors='red', edgecolors="black", linewidths=0.5, s=220)     #best optimized
        opt_points2 = ax2.scatter3D(X[-opt:-1,0],X[-opt:-1,1],Likelihood[-opt:-1], marker='o',c=Colors, edgecolors="black", linewidths=0.5, s=40)      #optimized trees
        
    elif paup_MAP:
        ax2.scatter3D(X[-3,0],X[-3,1],Likelihood[-3], marker='D',facecolors='red', edgecolors="black", linewidths=0.5, s=220)     #best optimized
        ax2.scatter3D(X[-2,0],X[-2,1],Likelihood[-2], marker='o',facecolors='white', edgecolors="black", linewidths=0.5, s=140)      #PAUP
        ax2.scatter3D(X[-1,0],X[-1,1],Likelihood[-1], marker='o',facecolors='black', edgecolors="black", linewidths=0.5, s=140)      #MAP
        opt_points2 = ax2.scatter3D(X[-opt:-2,0],X[-opt:-2,1],Likelihood[-opt:-2], marker='o',c=Colors, edgecolors="black", linewidths=0.5, s=40)      #optimized trees

        
    elif paup_RXML:
        ax2.scatter3D(X[-3,0],X[-3,1],Likelihood[-3], marker='D',facecolors='red', edgecolors="black", linewidths=0.5, s=220)     #best optimized
        ax2.scatter3D(X[-2,0],X[-2,1],Likelihood[-2], marker='o',facecolors='white', edgecolors="black", linewidths=0.5, s=140)      #PAUP
        ax2.scatter3D(X[-1,0],X[-1,1],Likelihood[-1], marker='o',facecolors='black', edgecolors="black", linewidths=0.5, s=140)      #RXML bropt
        opt_points2 = ax2.scatter3D(X[-opt:-2,0],X[-opt:-2,1],Likelihood[-opt:-2], marker='o',c=Colors, edgecolors="black", linewidths=0.5, s=40)      #optimized trees
        
    elif user_trees:
        nn= len(usertrees)
        ax2.scatter3D(X[-(nn+1),0],X[-(nn+1),1],Likelihood[-(nn+1)], marker='D',facecolors='red', edgecolors="black", linewidths=0.5, s=220)     #best optimized
        for i in range(1, nn+1):
            ax2.scatter3D(X[-i,0],X[-i,1],Likelihood[-i], marker='o',facecolors='white', edgecolors="black", linewidths=0.5, s=140)      #usertrees
        opt_points2 = ax2.scatter3D(X[-opt:-nn,0],X[-opt:-nn,1],Likelihood[-opt:-nn], marker='o',c=Colors, edgecolors="black", linewidths=0.5, s=40)      #optimized trees
        
    else:
        ax2.scatter3D(X[-1,0],X[-1,1],Likelihood[-1], marker='D',facecolors='red', edgecolors="black", linewidths=0.5, s=150)     #best optimized
        opt_points2 = ax2.scatter3D(X[-opt:,0],X[-opt:,1],Likelihood[-opt:], marker='o',c=Colors, edgecolors="black", linewidths=0.5, s=40)      #optimized trees
        
    ax2.grid(linestyle='-', linewidth='5', color='green')
    points.set_clim([np.min(Likelihood), np.max(Likelihood)])

    ax2.set_xlabel('Coordinate 1', labelpad=1,  fontsize=8)
    ax2.set_ylabel('Coordinate 2', labelpad=2,  fontsize=8)
    ax2.set_xlim(np.min(X[:,0]), np.max(X[:,0]))
    ax2.set_ylim(np.min(X[:,1]), np.max(X[:,1]))

    ax2.tick_params(labelsize=4.5 , pad=1)    #to change the size of numbers on axis & space between ticks and labels
    ax2.tick_params(axis='z',labelsize=5 , pad=4)    #to change the size of numbers on axis & space between ticks and labels
    
    ax2.ticklabel_format(useOffset=False, style='plain')   # to get rid of exponential format of numbers on all axis
#    ax2.view_init(20, -45)     #change the angle of 3D plot
    ax2.view_init(20, 135)     #change the angle of 3D plot


    v1 = np.linspace(np.min(Likelihood), np.max(Likelihood), 5, endpoint=True)
    print(f"v1 = {v1}")
    norm1= mpl.colors.Normalize(vmin=min(contour.cvalues.min(),np.min(Likelihood)), vmax=max(contour.cvalues.max(), np.max(Likelihood)))
    sm1 = plt.cm.ScalarMappable(norm=norm1, cmap = contour.cmap)
    sm1.set_array([])
    cbar = fig.colorbar(sm1, ax = ax2, shrink = 0.32, aspect = 12, pad=0.11, ticks=v1)
    cbar.ax.set_yticklabels(["{:4.4f}".format(i) for i in v1])    #To get rid of exponention expression of cbar numbers
    cbar.set_label('Log Likelihood', rotation=90 , labelpad=-75, fontsize=8)
    cbar.ax.tick_params(labelsize=6)   #to change the size of numbers on cbar
    cbar.outline.set_edgecolor('none')
    plt.ticklabel_format(useOffset=False)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if paup_MAP:
        ax3 = plt.subplot2grid((40,350), (29,92), rowspan=1, colspan=52, frameon=False)
    elif paup_RXML:
        ax3 = plt.subplot2grid((40,350), (29,92), rowspan=1, colspan=52, frameon=False)
    elif user_trees:
        ax3 = plt.subplot2grid((40,350), (29,92), rowspan=1, colspan=52, frameon=False)
    else:
        ax3 = plt.subplot2grid((40,350), (28,92), rowspan=1, colspan=52, frameon=False)

    if paup_tree:     #extract paup one
        Colors_bar = [colormap(i) for i in np.linspace(0.1, 0.9,N_topo+1)]
        bounds= Likelihood[-(opt):-1]
        bounds = [bounds[0]-5]+ bounds
    elif paup_MAP:     #extract paup &MAP
        Colors_bar = [colormap(i) for i in np.linspace(0.1, 0.9,N_topo+1)]
        bounds= Likelihood[-(opt):-2]
        bounds = [bounds[0]-5]+ bounds
    elif paup_RXML:     #extract paup &RXML
        Colors_bar = [colormap(i) for i in np.linspace(0.1, 0.9,N_topo+1)]
        bounds= Likelihood[-(opt):-2]
        bounds = [bounds[0]-5]+ bounds
    elif user_trees:     #extract user_trees
        Colors_bar = [colormap(i) for i in np.linspace(0.1, 0.9,N_topo+1)]
        bounds= Likelihood[-(opt):-len(usertrees)]
        bounds = [bounds[0]-5]+ bounds
    else:
        Colors_bar = [colormap(i) for i in np.linspace(0.1, 0.9,N_topo+1)]
        bounds= Likelihood[-(opt):]
        bounds = [bounds[0]-5]+ bounds

    cmap = mpl.colors.ListedColormap(Colors_bar)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    cb2 = mpl.colorbar.ColorbarBase(ax3, cmap=cmap, norm=norm, boundaries=bounds, ticks=bounds, spacing='uniform', orientation='horizontal')
    count=0
    for label in ax3.get_xticklabels():
        count +=1
        label.set_ha("right")
        label.set_rotation(40)

    ax3.tick_params(labelsize=4.5 , pad=1)    #to change the size of numbers on axis & space between ticks and labels
    ax3.set_xticklabels(["{:4.4f}".format(i) for i in bounds])    #To get rid of exponention expression of numbers
    cb2.outline.set_edgecolor('white')   #colorbar externals edge color

    labels = ax3.get_xticklabels()
    len_label= len(labels)

    r=5
    if len(labels)>10:
        v2 = np.linspace(0,len_label, r,  dtype= int, endpoint=False)
        for i in range(len_label-1):
            if i not in v2:
                labels[i] = ""
    labels[0] = ""
    ax3.set_xticklabels(labels)
    #OR:
    for label in ax3.get_xticklabels():    #remove all labels
        label.set_visible(False)
        
    ax3.tick_params(size=0) #Remove all ticks
    ax3.set_title("Topology spectrum of {} optimized trees \ncorresponding to their LogLike values ".format(N_topo), loc='center', pad=7, fontsize=6, fontweight ="bold")

    plt.subplots_adjust(left=0.05, bottom=-0.25, right=0.98, top=1.3, wspace=None, hspace=None)  #to adjust white spaces around figure
    plt.savefig(filename, format='pdf')
    plt.show()
    
  
    if validation_mds:
        MDS_distances, mds_validation_values = MDS_validation(M,X,1,it+1)
#        np.savetxt ('MDS_distances', MDS_distances,  fmt='%s')
#        np.savetxt ('X_MDS_coordinates', X,  fmt='%s')
#        np.savetxt ('Real_distances', M,  fmt='%s')


#=================== Interpolation method2: RBF (Radial basis function interpolation) (Contour & Surface) ====================
def point_in_hull(point, hull, tolerance=1e-12):
    return all(
        (np.dot(eq[:-1], point) + eq[-1] <= tolerance)
        for eq in hull.equations)
        
        
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def interpolate_rbf(it, filename, M, Likelihood, bestlike, Treelist, StartTrees,  optimized_BestTrees, Topologies,NUMPATHTREES, dataset_option_list,dataset_option, smoothness, validation_mds, hull_indices=None):

    paup_tree = dataset_option_list[0]
    paup_MAP = dataset_option_list[1]
    paup_RXML = dataset_option_list[2]
    user_trees = dataset_option_list[3]
    

    n = len(Treelist)
    N = len(StartTrees)
    n_path = NUMPATHTREES-2
    if paup_tree:
        opt = len(optimized_BestTrees)+1    #just PAUP
    elif paup_MAP:
        opt = len(optimized_BestTrees)+2    #both PAUP&MAP
    elif paup_RXML:
        opt = len(optimized_BestTrees)+2    #both PAUP&RXML
    elif user_trees:
        with open(dataset_option,'r') as myfile:
            usertrees = myfile.readlines()
        opt = len(optimized_BestTrees)+len(usertrees)    #user_trees
    else:
        opt = len(optimized_BestTrees)
    
    all_path = n-opt-N
    num=200
    
    X= MDS(M,2)
    xx = np.linspace(np.min(X),np.max(X),num)
    yy =  np.linspace(np.min(X),np.max(X),num)
    XX, YY = np.meshgrid(xx,yy)          #We DONT know the likelihood values of these points
    
    
    interp_types = ['multiquadric', 'inverse', 'gaussian', 'linear', 'cubic',
                'quintic', 'thin_plate']
#    Test:
#    smooth_list=[0.0, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 2, 10]
#    smooth_list=[0.0, 1e-10]
#    smooth_list=[1e-10]
    smooth_list=[smoothness]
    for s_num,s1 in enumerate(smooth_list):
        rbfi = Rbf(np.real(X[:,0]), np.real(X[:,1]), Likelihood, smooth=s1, function = interp_types[-1])
        values = rbfi(XX, YY)   # interpolated values(likelihods)
        hull = ConvexHull(X)
        coordinate_grid = np.array([XX, YY])
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                p= coordinate_grid[:, i, j]
                point_is_in_hull = point_in_hull(p, hull)
                if not(point_is_in_hull):
                    values[i][j] = np.nan
                pass
            pass


        N_topo = len(Topologies)
        colormap = plt.cm.RdPu
        Colors = [colormap(i) for i in np.linspace(0.1, 0.9,N_topo)]

        fig = plt.figure(figsize=(12,5))
        fig.set_facecolor('white')
        plt.rcParams['grid.color'] = "white" # change color
        plt.rcParams['grid.linewidth'] = 1   # change linwidth
        plt.rcParams['grid.linestyle'] = '-'

        ax1 = plt.subplot2grid((40,350), (14,0),rowspan=13, colspan=80)
        contour = ax1.contourf(XX,YY,values, 10, alpha=1, vmin=np.nanmin(values), vmax=np.nanmax(values), cmap='viridis')

        vmin, vmax = np.nanmin(values.flatten()), np.nanmax(values.flatten())
        ax1.scatter(X[:N,0], X[:N,1], alpha=1, marker='^',facecolors='none', vmin = vmin, vmax = vmax, cmap='viridis',  edgecolors="black", linewidths=0.7, s=60)     #starttrees
        points = ax1.scatter(X[N:,0], X[N:,1], c=Likelihood[N:] , alpha=1,  marker='o', cmap='viridis',  edgecolors="black", linewidths=0.3, s=5)  #PathTrees
        
        if paup_tree:
            ax1.scatter(X[-1,0],X[-1,1], marker='o',facecolors='white', edgecolors="black", linewidths=0.5, s=140)      #PAUP
            ax1.scatter(X[-2,0],X[-2,1], marker='D', facecolors='red', edgecolors="black", linewidths=0.5, s=220)     #best optimized
            opt_points = ax1.scatter(X[-opt:-1,0],X[-opt:-1,1], marker='o',c=Colors, edgecolors="black", linewidths=0.5, s=40)      #optimized trees
        
        elif paup_MAP:
            ax1.scatter(X[-3,0],X[-3,1], marker='D', facecolors='red', edgecolors="black", linewidths=0.5, s=220)     #best optimized
            ax1.scatter(X[-2,0],X[-2,1], marker='o',facecolors='white', edgecolors="black", linewidths=0.5, s=140)      #PAUP
            ax1.scatter(X[-1,0],X[-1,1], marker='o', facecolors='black', edgecolors="black", linewidths=0.5, s=140)     #MAP
            opt_points = ax1.scatter(X[-opt:-2,0],X[-opt:-2,1], marker='o',c=Colors, edgecolors="black", linewidths=0.5, s=40)      #optimized trees
            
        elif paup_RXML:
            ax1.scatter(X[-3,0],X[-3,1], marker='D', facecolors='red', edgecolors="black", linewidths=0.5, s=220)     #best optimized
            ax1.scatter(X[-2,0],X[-2,1], marker='o',facecolors='white', edgecolors="black", linewidths=0.5, s=140)      #PAUP
            ax1.scatter(X[-1,0],X[-1,1], marker='o', facecolors='black', edgecolors="black", linewidths=0.5, s=140)     #RXML_bropt
            opt_points = ax1.scatter(X[-opt:-2,0],X[-opt:-2,1], marker='o',c=Colors, edgecolors="black", linewidths=0.5, s=40)      #optimized trees
            
        elif user_trees:
            nn= len(usertrees)
            ax1.scatter(X[-(nn+1),0],X[-(nn+1),1], marker='D', facecolors='red', edgecolors="black", linewidths=0.5, s=220)     #best optimized
            for i in range(1, nn+1):
                ax1.scatter(X[-i,0],X[-i,1], marker='o',facecolors='white', edgecolors="black", linewidths=0.5, s=140)      #usertrees
            opt_points = ax1.scatter(X[-opt:-nn,0],X[-opt:-nn,1], marker='o',c=Colors, edgecolors="black", linewidths=0.5, s=40)      #optimized trees
        else:
            ax1.scatter(X[-1,0],X[-1,1], marker='D', facecolors='red', edgecolors="black", linewidths=0.5, s=200)     #best optimized
            opt_points = ax1.scatter(X[-opt:,0],X[-opt:,1], marker='o',c=Colors, edgecolors="black", linewidths=0.5, s=40)      #optimized trees
            
        for j, (top, c) in enumerate(zip(Topologies, Colors)):     #topologies trees
            ax1.scatter(X[top,0], X[top,1], marker='o', color=c, edgecolors="black", linewidths=0.15,s=8)
        
      
        min_xvalue, max_xvalue = np.min(X[:,0]),np.max(X[:,0])       #To hadle the distance between boundary and contour
        min_yvalue, max_yvalue = np.min(X[:,1]),np.max(X[:,1])
        x_dist =(max_xvalue-min_xvalue)/30
        y_dist =(max_yvalue-min_yvalue)/30
        
        ax1.grid(linestyle=':', linewidth='0.2', color='white', alpha=0.6)
        ax1.set_xlim(min_xvalue-x_dist, max_xvalue+x_dist)
        ax1.set_ylim(min_yvalue-y_dist, max_yvalue+y_dist)
        
        ax1.set_xlabel('Coordinate 1', labelpad=7, fontsize=8)
        ax1.set_ylabel('Coordinate 2', labelpad=3, fontsize=8)
        ax1.tick_params(labelsize=4.5)

        ax1.ticklabel_format(useOffset=False, style='plain')
        fig.autofmt_xdate()    #to fix the issue of overlapping x-axis labels
        

        if n_path==1:
            smallgreen_circle = mlines.Line2D([], [], color='limegreen', marker='o', linestyle='None',mec="black", mew=0.3, markersize=3, label="{} pathtrees + {} starting trees:\n{}-tree between each pair".format(all_path,N, n_path))
        else:
            smallgreen_circle = mlines.Line2D([], [], color='limegreen', marker='o', linestyle='None',mec="black", mew=0.3, markersize=3, label="{} pathtrees + {} starting trees:\n{}-trees between each pair".format(all_path,N, n_path))
        green_triangle = mlines.Line2D([], [], color='none', marker='^', linestyle='None',mec="black", mew=0.7, markersize=8, label="{} starting trees".format(len(StartTrees)))
        smallpink_circle = mlines.Line2D([], [], color='hotpink', marker='o', linestyle='None', mec="black", mew=0.25, markersize=3, label="{} trees for topology analysis ".format(len(bestlike)))
        bigpink_circle = mlines.Line2D([], [], color='hotpink', marker='o', linestyle='None',mec="black", mew=0.4, markersize=6, label="{} optimized trees  from different \ntopologies".format(N_topo))
        
        
        if paup_tree:
            bigred_square = mlines.Line2D([], [], color='red', marker='D', linestyle='None',mec="black", mew=0.7, markersize=11, label="PATHTREES best optimized tree \nLogLike {:4.4f}".format(Likelihood[-2]))
            bigwhite_circle = mlines.Line2D([], [],  marker='o', linestyle='None',mec="black",mfc='white', mew=0.9, markersize=11, label="PAUP best optimized tree \nLogLike {:4.4f}".format(Likelihood[-1]))
            plt.legend(bbox_to_anchor=(1.05, 0.95),  fontsize='x-small', labelspacing = 1.5, frameon=False, scatterpoints=1, handles=[ smallgreen_circle, green_triangle, smallpink_circle, bigpink_circle,   bigred_square, bigwhite_circle])
        
        elif paup_MAP:
            bigred_square = mlines.Line2D([], [], color='red', marker='D', linestyle='None',mec="black", mew=0.7, markersize=11, label="PATHTREES best optimized tree \nLogLike {:4.4f}".format(Likelihood[-3]))
            bigwhite_circle = mlines.Line2D([], [],  marker='o', linestyle='None',mec="black",mfc='white', mew=0.9, markersize=11, label="PAUP best optimized tree \nLogLike {:4.4f}".format(Likelihood[-2]))
            bigblack_circle = mlines.Line2D([], [],  marker='o', linestyle='None',mec="black",mfc='black', mew=0.9, markersize=11, label="MAP best optimized tree \nLogLike {:4.4f}".format(Likelihood[-1]))
            plt.legend(bbox_to_anchor=(1.05, 0.95),  fontsize='x-small', labelspacing = 1.5, frameon=False, scatterpoints=1, handles=[ smallgreen_circle, green_triangle, smallpink_circle, bigpink_circle,   bigred_square, bigwhite_circle, bigblack_circle])
            
            
        elif paup_RXML:
            bigred_square = mlines.Line2D([], [], color='red', marker='D', linestyle='None',mec="black", mew=0.7, markersize=11, label="PATHTREES best optimized tree \nLogLike {:4.4f}".format(Likelihood[-3]))
            bigwhite_circle = mlines.Line2D([], [],  marker='o', linestyle='None',mec="black",mfc='white', mew=0.9, markersize=11, label="PAUP best optimized tree \nLogLike {:4.4f}".format(Likelihood[-2]))
            bigblack_circle = mlines.Line2D([], [],  marker='o', linestyle='None',mec="black",mfc='black', mew=0.9, markersize=11, label="RAxML best optimized tree \nLogLike {:4.4f}".format(Likelihood[-1]))
            plt.legend(bbox_to_anchor=(1.05, 0.95),  fontsize='x-small', labelspacing = 1.5, frameon=False, scatterpoints=1, handles=[ smallgreen_circle, green_triangle, smallpink_circle, bigpink_circle,   bigred_square, bigwhite_circle, bigblack_circle])
            
        elif user_trees:
            nn= len(usertrees)
            bigred_square = mlines.Line2D([], [], color='red', marker='D', linestyle='None',mec="black", mew=0.7, markersize=11, label="PATHTREES best optimized tree \nLogLike {:4.4f}".format(Likelihood[-(nn+1)]))
            bigwhite_circle = mlines.Line2D([], [],  marker='o', linestyle='None',mec="black",mfc='white', mew=0.9, markersize=11, label="user trees")
            plt.legend(bbox_to_anchor=(1.05, 0.85),  fontsize='x-small', labelspacing = 1.5, frameon=False, scatterpoints=1, handles=[ smallgreen_circle, green_triangle, smallpink_circle, bigpink_circle,   bigred_square, bigwhite_circle])
        
        else:
            bigred_square = mlines.Line2D([], [], color='red', marker='D', linestyle='None',mec="black", mew=0.7, markersize=11, label="PATHTREES best optimized tree \nLogLike {:4.4f}".format(Likelihood[-1]))
            plt.legend(bbox_to_anchor=(1.05, 0.85),  fontsize='x-small', labelspacing = 1.5, frameon=False, scatterpoints=1, handles=[ smallgreen_circle, green_triangle, smallpink_circle, bigpink_circle,   bigred_square])
        plt.setp(ax1.spines.values(), color='gray')

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ax2 = plt.subplot2grid((40,350), (0,150), rowspan=50, colspan=200, projection='3d')
        ax2.plot_wireframe(XX, YY, values,  rstride=2, cstride=2, linewidth=1,alpha=0.3, cmap='viridis')
        surf = ax2.plot_surface(XX, YY, values,  rstride=1, cstride=1, edgecolor='none',alpha=0.5, cmap='viridis', vmin=np.nanmin(values), vmax=np.nanmax(values))

        min_value, max_value = ax2.get_zlim()       #To hadle the distance between surface and contour

        cset = ax2.contourf(XX, YY, values, vmin=np.nanmin(values), vmax=np.nanmax(values), zdir='z', offset=min_value, cmap='viridis', alpha=0.3)

        ax2.scatter3D(X[:N,0], X[:N,1], Likelihood[:N], marker='^',facecolors='none',  vmin = vmin, vmax = vmax, edgecolors="black", linewidths=0.5, s=50)    #starttrees
        
        if paup_tree:
            ax2.scatter3D(X[-1,0],X[-1,1],Likelihood[-1], marker='o',facecolors='white', edgecolors="black", linewidths=0.5, s=140)      #PAUP
            ax2.scatter3D(X[-2,0],X[-2,1],Likelihood[-2], marker='D',facecolors='red', edgecolors="black", linewidths=0.5, s=220)     #best optimized
            opt_points2 = ax2.scatter3D(X[-opt:-1,0],X[-opt:-1,1],Likelihood[-opt:-1], marker='o',c=Colors, edgecolors="black", linewidths=0.5, s=40)      #optimized trees
            
        elif paup_MAP:
            ax2.scatter3D(X[-3,0],X[-3,1],Likelihood[-3], marker='D',facecolors='red', edgecolors="black", linewidths=0.5, s=220)     #best optimized
            ax2.scatter3D(X[-2,0],X[-2,1],Likelihood[-2], marker='o',facecolors='white', edgecolors="black", linewidths=0.5, s=140)      #PAUP
            ax2.scatter3D(X[-1,0],X[-1,1],Likelihood[-1], marker='o',facecolors='black', edgecolors="black", linewidths=0.5, s=140)      #MAP
            opt_points2 = ax2.scatter3D(X[-opt:-2,0],X[-opt:-2,1],Likelihood[-opt:-2], marker='o',c=Colors, edgecolors="black", linewidths=0.5, s=40)      #optimized trees
            
        elif paup_RXML:
            ax2.scatter3D(X[-3,0],X[-3,1],Likelihood[-3], marker='D',facecolors='red', edgecolors="black", linewidths=0.5, s=220)     #best optimized
            ax2.scatter3D(X[-2,0],X[-2,1],Likelihood[-2], marker='o',facecolors='white', edgecolors="black", linewidths=0.5, s=140)      #PAUP
            ax2.scatter3D(X[-1,0],X[-1,1],Likelihood[-1], marker='o',facecolors='black', edgecolors="black", linewidths=0.5, s=140)      #RXML bropt
            opt_points2 = ax2.scatter3D(X[-opt:-2,0],X[-opt:-2,1],Likelihood[-opt:-2], marker='o',c=Colors, edgecolors="black", linewidths=0.5, s=40)      #optimized trees
            
        elif user_trees:
            nn= len(usertrees)
            ax2.scatter3D(X[-(nn+1),0],X[-(nn+1),1],Likelihood[-(nn+1)], marker='D',facecolors='red', edgecolors="black", linewidths=0.5, s=220)     #best optimized
            for i in range(1, nn+1):
                ax2.scatter3D(X[-i,0],X[-i,1],Likelihood[-i], marker='o',facecolors='white', edgecolors="black", linewidths=0.5, s=140)      #usertrees
            opt_points2 = ax2.scatter3D(X[-opt:-nn,0],X[-opt:-nn,1],Likelihood[-opt:-nn], marker='o',c=Colors, edgecolors="black", linewidths=0.5, s=40)      #optimized trees
            
        else:
            ax2.scatter3D(X[-1,0],X[-1,1],Likelihood[-1], marker='D',facecolors='red', edgecolors="black", linewidths=0.5, s=150)     #best optimized
            opt_points2 = ax2.scatter3D(X[-opt:,0],X[-opt:,1],Likelihood[-opt:], marker='o',c=Colors, edgecolors="black", linewidths=0.5, s=40)      #optimized trees
            
        ax2.grid(linestyle='-', linewidth='5', color='green')
        points.set_clim([np.min(Likelihood), np.max(Likelihood)])

        
        ax2.set_xlabel('Coordinate 1', labelpad=1,  fontsize=8)
        ax2.set_ylabel('Coordinate 2', labelpad=2,  fontsize=8)
        ax2.set_xlim(np.min(X[:,0]), np.max(X[:,0]))
        ax2.set_ylim(np.min(X[:,1]), np.max(X[:,1]))

        ax2.tick_params(labelsize=4.5 , pad=1)    #to change the size of numbers on axis & space between ticks and labels
        ax2.tick_params(axis='z',labelsize=5 , pad=4)    #to change the size of numbers on axis & space between ticks and labels
        
        ax2.ticklabel_format(useOffset=False, style='plain')   # to get rid of exponential format of numbers on all axis
#        ax2.view_init(20, -45)     #change the angle of 3D plot
        ax2.view_init(20, 135)     #change the angle of 3D plot


        v1 = np.linspace(np.min(Likelihood), np.max(Likelihood), 5, endpoint=True)
        norm1= mpl.colors.Normalize(vmin=min(contour.cvalues.min(),np.min(Likelihood)), vmax=max(contour.cvalues.max(), np.max(Likelihood)))
        sm1 = plt.cm.ScalarMappable(norm=norm1, cmap = contour.cmap)
        sm1.set_array([])
        cbar = fig.colorbar(sm1, ax = ax2, shrink = 0.32, aspect = 12, pad=0.11, ticks=v1)
        cbar.ax.set_yticklabels(["{:4.4f}".format(i) for i in v1])    #To get rid of exponention expression of cbar numbers
        cbar.set_label('Log Likelihood', rotation=90 , labelpad=-75, fontsize=8)
        cbar.ax.tick_params(labelsize=6)   #to change the size of numbers on cbar
        cbar.outline.set_edgecolor('none')
        plt.ticklabel_format(useOffset=False)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if paup_MAP:
            ax3 = plt.subplot2grid((40,350), (29,92), rowspan=1, colspan=52, frameon=False)
        elif paup_RXML:
            ax3 = plt.subplot2grid((40,350), (29,92), rowspan=1, colspan=52, frameon=False)
        elif user_trees:
            ax3 = plt.subplot2grid((40,350), (29,92), rowspan=1, colspan=52, frameon=False)
        else:
            ax3 = plt.subplot2grid((40,350), (28,92), rowspan=1, colspan=52, frameon=False)

        if paup_tree:     #extract paup one
            Colors_bar = [colormap(i) for i in np.linspace(0.1, 0.9,N_topo+1)]
            bounds= Likelihood[-(opt):-1]
            bounds = [bounds[0]-5]+ bounds
        elif paup_MAP:     #extract paup &MAP
            Colors_bar = [colormap(i) for i in np.linspace(0.1, 0.9,N_topo+1)]
            bounds= Likelihood[-(opt):-2]
            bounds = [bounds[0]-5]+ bounds
        elif paup_RXML:     #extract paup &RXML
            Colors_bar = [colormap(i) for i in np.linspace(0.1, 0.9,N_topo+1)]
            bounds= Likelihood[-(opt):-2]
            bounds = [bounds[0]-5]+ bounds
        elif user_trees:     #extract user_trees
            Colors_bar = [colormap(i) for i in np.linspace(0.1, 0.9,N_topo+1)]
            bounds= Likelihood[-(opt):-len(usertrees)]
            bounds = [bounds[0]-5]+ bounds
        else:
            Colors_bar = [colormap(i) for i in np.linspace(0.1, 0.9,N_topo+1)]
            bounds= Likelihood[-(opt):]
            bounds = [bounds[0]-5]+ bounds

        cmap = mpl.colors.ListedColormap(Colors_bar)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        cb2 = mpl.colorbar.ColorbarBase(ax3, cmap=cmap, norm=norm, boundaries=bounds, ticks=bounds, spacing='uniform', orientation='horizontal')
        count=0
        for label in ax3.get_xticklabels():
            count +=1
            label.set_ha("right")
            label.set_rotation(40)

        ax3.tick_params(labelsize=4.5 , pad=1)    #to change the size of numbers on axis & space between ticks and labels
        ax3.set_xticklabels(["{:4.4f}".format(i) for i in bounds])    #To get rid of exponention expression of numbers
        cb2.outline.set_edgecolor('white')   #colorbar externals edge color

        labels = ax3.get_xticklabels()
        len_label= len(labels)

        r=5
        if len(labels)>10:
            v2 = np.linspace(0,len_label, r,  dtype= int, endpoint=False)
            for i in range(len_label-1):
                if i not in v2:
                    labels[i] = ""
        labels[0] = ""
        ax3.set_xticklabels(labels)
        #OR:
        for label in ax3.get_xticklabels():    #remove all labels
            label.set_visible(False)
            
        ax3.tick_params(size=0) #Remove all ticks
        ax3.set_title("Topology spectrum of {} optimized trees \ncorresponding to their LogLike values ".format(N_topo), loc='center', pad=7, fontsize=6, fontweight ="bold")

        plt.subplots_adjust(left=0.05, bottom=-0.25, right=0.98, top=1.3, wspace=None, hspace=None)  #to adjust white spaces around figure
        if len(smooth_list)==1:
            plt.savefig(filename, format='pdf')
        elif len(smooth_list)>1:
            plt.savefig(str(s1)+filename, format='pdf')
        plt.show()


    if validation_mds:
        MDS_distances, mds_validation_values = MDS_validation(M,X,1,it+1)
#        np.savetxt ('MDS_distances', MDS_distances,  fmt='%s')
#        np.savetxt ('X_MDS_coordinates', X,  fmt='%s')
#        np.savetxt ('Real_distances', M,  fmt='%s')

#=========================================== Convex Hull: (default griddata cubic) ===================================================
def boundary_convexhull(M,Likelihood,treelist, iter_num):
    meth= 'cubic'
    num=100
    X= MDS(M,2)
    xx = np.linspace(np.min(X),np.max(X),num)
    yy =  np.linspace(np.min(X),np.max(X),num)
    XX, YY = np.meshgrid(xx,yy)          #We DONT know the likelihood values of these points
    values = griddata((np.real(X[:,0]), np.real(X[:,1])), Likelihood, (XX, YY), method=meth)
    
    fig = plt.figure(figsize=(12,5))
    fig.set_facecolor('white')
    plt.rcParams['grid.color'] = "white" # change color
    plt.rcParams['grid.linewidth'] = 1   # change linwidth
    plt.rcParams['grid.linestyle'] = '-'
    
    ax1 = plt.subplot2grid((20,40), (3,0),rowspan=14, colspan=12)
    contour = ax1.contourf(XX, YY, values, alpha=0.75, vmin=np.nanmin(values), vmax=np.nanmax(values))
    points=ax1.scatter(X[:,0], X[:,1], alpha=1,  marker='o', c=Likelihood , cmap='viridis',  edgecolors="black", linewidths=0.001, s=14)  #Trees
    
    ax1.grid(linestyle=':', linewidth='0.2', color='white', alpha=0.6)
    ax1.set_xlim(np.min(X[:,0])-0.01, np.max(X[:,0])+0.01)
    ax1.set_ylim(np.min(X[:,1])-0.01, np.max(X[:,1])+0.01)
    ax1.set_xlabel('Coordinate 1', labelpad=3, fontsize=9)
    ax1.set_ylabel('Coordinate 2', labelpad=3, fontsize=9)
    ax1.tick_params(labelsize=4.5)
    ax1.ticklabel_format(useOffset=False, style='plain')
    
    
    hull = ConvexHull(X)
    # Get the indices of the hull points.
    hull_indices = hull.vertices
    print(f"    Number of Boundary trees = {len(hull_indices)}\n")
    # These are the actual points.
    hull_pts = X[hull_indices, :]
    Boundary_Trees = [treelist[i] for i in hull_indices]
    Boundary_Trees = [s.replace('\n', '') for s in Boundary_Trees]
#    np.savetxt ('Boundary_Trees', Boundary_Trees,  fmt='%s')
#    print("\n\nBoundary_Trees =\n",Boundary_Trees)
    
    ax2 = plt.subplot2grid((20,40), (3,16), rowspan=14, colspan=16)
    v1 = np.linspace(np.min(Likelihood), np.max(Likelihood), 7, endpoint=True)

    norm1= mpl.colors.Normalize(vmin=min(contour.cvalues.min(),np.min(Likelihood)), vmax=max(contour.cvalues.max(), np.max(Likelihood)))
    sm1 = plt.cm.ScalarMappable(norm=norm1, cmap = contour.cmap)
    sm1.set_array([])
    cbar = fig.colorbar(sm1, ax = ax2, shrink = 1, aspect = 15, pad=0.07, ticks=v1)
    
    ax2.scatter(X[:,0], X[:,1],alpha=1, marker='o', c=Likelihood, cmap='viridis',  s=6)

    ax2.scatter(hull_pts[:,0], hull_pts[:,1], marker='o', c='r',  s=40)
    plt.fill(hull_pts[:,0], hull_pts[:,1], fill=False, facecolor='none', edgecolor='r',  linewidth=1, ls='--')
    cbar.set_label('Log Likelihood', rotation=90 , labelpad=10, fontsize=10)     #Dec27.2022
    ax2.set_xlabel('Coordinate 1', labelpad=3,  fontsize=9)
    ax2.set_ylabel('Coordinate 2', labelpad=3,  fontsize=9)
    ax2.set_xlim(np.min(X[:,0])-0.01, np.max(X[:,0])+0.01)
    ax2.set_ylim(np.min(X[:,1])-0.01, np.max(X[:,1])+0.01)
    ax2.tick_params(labelsize=4.5 , pad=1)    #to change the size of numbers on axis & space between ticks and labels
    ax2.ticklabel_format(useOffset=False, style='plain')   # to get rid of exponential format of numbers on all axis
    cbar.ax.tick_params(labelsize=6)   #to change the size of numbers on cbar
    cbar.outline.set_edgecolor('none')
#    fig.autofmt_xdate()    #to fix the issue of overlapping x-axis labels

    plt.subplots_adjust(left=0.05, bottom=-0.1, right=1.2, top=1.1, wspace=None, hspace=None)  #to adjust hite spaces around figure
    plt.savefig('Boundary_iter{}.pdf'.format(iter_num), format='pdf')
#    plt.show()
    plt.close(fig)
    return Boundary_Trees







if __name__ == "__main__":
        print(f"TEST")

