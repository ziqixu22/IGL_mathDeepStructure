import numpy as np
import pandas as pd
from Bio import AlignIO
from Bio import SeqIO
from Bio import Seq
import numpy as np
import os

df_ref = pd.read_csv(r"C:\Users\praje\OneDrive\Documents\PathTrees\PATHTREES\data\references_data.csv", header=None, names=['col1', 'col2', 'col3'])
df_ref = pd.DataFrame(df_ref)


output_handle = r"C:\Users\praje\OneDrive\Documents\PathTrees\PATHTREES\Matrix\Ref.phy"



max_seq_len = df_ref.col3.str.len().max()


# Convert the DataFrame into a dictionary of sequence
sequences = {}
for i in range(len(df_ref.col3)):
    sequence = df_ref.col3.iat[i]

    padded_sequence = sequence.ljust(max_seq_len, '-') 

    #sequences[df_ref.col2.iat[i]] = padded_sequence
    n = df_ref.col2.iat[i][:10]
    n = n.encode('ascii', 'ignore').decode('ascii')
    sequences[n] = padded_sequence
    
        
# Specify the sequence type (e.g., 'protein' for amino acid sequences)
sequence_type = 'references'

# Write the sequences to a PHYLIP format file
phylip_file = r"C:\Users\praje\OneDrive\Documents\PathTrees\PATHTREES\Matrix\Ref.phy" # Replace with your desired output file name
with open(phylip_file, 'w') as f:
    # Write the number of sequences and sequence length in the first line
    f.write(f"{len(sequences)} {max_seq_len}\n")

    # Write sequences with their names
    for name, sequence in sequences.items():
        f.write(f"{name} {sequence}\n")