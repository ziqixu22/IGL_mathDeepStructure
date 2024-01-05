import sys
import numpy as np
from pathlib import Path

# Resolve the path of the current script and add its parent directory to the system path
file = Path(__file__).resolve()
parent = file.parent
sys.path.append(str(file.parent))

# Import custom modules; subtree for tree operations and copy for object copying
import subtree
import copy

# Flag for enabling debug mode and setting precision for numerical operations
DEBUG = False
precision = 10

def internalpathtrees(treefile, terminallist, numpathtrees):
    """
    Generates a path between two trees using geodesic principles.
    """

    # Extract subtrees and corresponding dictionaries from input file and terminal list
    Results_subtrees, tip_dict, edge_dict = subtree.subtrees(treefile, terminallist)
    T1 = Results_subtrees[0]  # First main tree
    T2 = Results_subtrees[1]  # Second main tree
    disjoint_indices = Results_subtrees[2]  # Indices of disjoint subtrees

    # Debug prints for visualizing trees and disjoint indices
    if DEBUG:
        print("T1", T1)
        print("\nTree1 subtrees :\n", T1)
        print("\n\nTree2 subtrees :\n", T2)
        print("\n\ndisjoint_indices :\n", disjoint_indices)

    # Dictionaries for tips and edges for both trees
    T1_tip_dict = tip_dict[0]
    T1_edge_dict = edge_dict[0]
    T2_tip_dict = tip_dict[1]
    T2_edge_dict = edge_dict[1]

    # Read the tree file and preprocess its content
    with open(treefile, "r") as File:
        file = File.readlines()
    treelist = np.array([s.replace('\n', '') for s in file])

    # Debug print for path tree creation
    if DEBUG:
        print(f'\n++++++++++++++++++    Create PathTrees   ++++++++++++++++++\n')

    # Generate a range of values between 0 and 1 for path tree interpolation
    Lamda = np.linspace(0.0, 1.0, num=numpathtrees)
    Lamda = [round(elem, precision) for elem in Lamda]
    thetreelist = []

    # Iterate over each lambda value to create interpolated path trees
    for l, lamda in enumerate(Lamda[1:-1]):
        if DEBUG:
            print(f'\n=================  PathTree #{l}, lamda={lamda} ================\n')

        T1_path = copy.deepcopy(T1)  # Deep copy of the first tree

        # Iterate over subtrees to interpolate between T1 and T2
        for num in range(len(T1_path)):
            if num in disjoint_indices[0]:  # Handling disjoint subtrees
                if DEBUG:
                    print(f'\n~~~~~~~~~~~~~~~~ subtree #{num}    ,    lamda{lamda} ~~~~~~~~~~~~~~~~~~\n')
                lambda_limits, epsilon = subtree.path_legs(num, T1, T2, T1_edge_dict, T2_edge_dict)
                edited_subtree = subtree.pathtree_edges(num, T1, T2, T1_edge_dict, T2_edge_dict, lamda, lambda_limits, epsilon)
                T1_path[num] = edited_subtree
            else:  # Handling common subtrees
                if DEBUG:
                    print(f'\n~~~~~~~~~~~~ common subtree num #{num}    ,    lamda{lamda} ~~~~~~~~~~~~~~\n')

                # Interpolating tip lengths and edge lengths
                T1_path[num][-2] = list((1 - lamda) * np.array(T1[num][-2]) + lamda * np.array(T2[num][-2]))
                T1_path[num][-2] = [round(elem, precision) for elem in T1_path[num][-2]]
                T1_path[num][-1] = list((1 - lamda) * np.array(T1[num][-1]) + lamda * np.array(T2[num][-1]))
                T1_path[num][-1] = [round(elem, precision) for elem in T1_path[num][-1]]

        if DEBUG:
            print(f'\n~~~~~~~~~~~~~~~~~~ generated pathtree #{l} ~~~~~~~~~~~~~~~~~~~~~\n')

        # Generate Newick format strings for the path trees
        sub_newicks, newick = subtree.Sub_Newicks(T1_path, disjoint_indices[0])
        thetreelist.append(newick)

    return thetreelist

# Main execution block
if __name__ == "__main__":
    # Check for correct number of command-line arguments
    if len(sys.argv) < 2:
        print("pathtrees.py treefile terminalist")

    # Extract arguments and call the function
    treefile = sys.argv[1]
    terminallist = sys.argv[2]
    numpathtrees = 10
    mypathtrees = internalpathtrees(treefile, terminallist, numpathtrees)
    print("Standalone test:\nlook at", mypathtrees)