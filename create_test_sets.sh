
# Updated Supervised Test Set. pinf test seqs + new labels + new vocab
python make_dataset_from_swissprot.py --data-file-path /home/samirchar/ProteinFunctions/data/swissprot/uniprot_sprot_may_2024.dat --output-file-path /home/samirchar/ProteinFunctions/data/swissprot/proteinfer_splits/random/test_GO_may_2024.fasta --sequence-vocabulary=proteinfer_test --label-vocabulary=all --parenthood-file-path /home/samirchar/ProteinFunctions/data/vocabularies/parenthood_may_2024.json

# Updated Supervised Test Set. pinf test seqs + new labels + pinf/old vocab
python make_dataset_from_swissprot.py --data-file-path /home/samirchar/ProteinFunctions/data/swissprot/uniprot_sprot_may_2024.dat --output-file-path /home/samirchar/ProteinFunctions/data/swissprot/proteinfer_splits/random/test_GO_may_2024_pinf_vocab.fasta --sequence-vocabulary=proteinfer_test --label-vocabulary=proteinfer --parenthood-file-path /home/samirchar/ProteinFunctions/data/vocabularies/parenthood_may_2024.json

# GO Zero Shot 2024 Leaf Nodes. new seqs + new labels only leaf nodes + only added vocab terms
python make_dataset_from_swissprot.py --data-file-path /home/samirchar/ProteinFunctions/data/swissprot/uniprot_sprot_may_2024.dat --output-file-path /home/samirchar/ProteinFunctions/data/zero_shot/GO_swissprot_leaf_nodes_may_2024.fasta --sequence-vocabulary=new --only-leaf-nodes --label-vocabulary=new --parenthood-file-path /home/samirchar/ProteinFunctions/data/vocabularies/parenthood_may_2024.json

# GO Zero Shot 2024 new seqs + new labels + only added vocab terms
python make_dataset_from_swissprot.py --data-file-path /home/samirchar/ProteinFunctions/data/swissprot/uniprot_sprot_may_2024.dat --output-file-path /home/samirchar/ProteinFunctions/data/zero_shot/GO_swissprot_may_2024.fasta --sequence-vocabulary=new --label-vocabulary=new --parenthood-file-path /home/samirchar/ProteinFunctions/data/vocabularies/parenthood_may_2024.json