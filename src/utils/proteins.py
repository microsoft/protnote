
AMINO_ACID_VOCABULARY = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R',
    'S', 'T', 'V', 'W', 'Y'
]
# Map AA characters to their index in VOCAB.
RESIDUE_TO_INT = {aa: idx for idx, aa in enumerate(AMINO_ACID_VOCABULARY)}
INT_TO_RESIDUE = {idx:aa for aa,idx in RESIDUE_TO_INT.items()}
