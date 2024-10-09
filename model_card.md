
# Model Card for ProtNote

<!-- Provide a quick summary of what the model is/does. -->

ProtNote is a multimodal deep learning model that leverages free-form text to enable both supervised and zero-shot protein function prediction

## Model Details

### Model Description

Understanding the protein sequence-function relationship is essential for advancing protein biology and engineering. However fewer than 1% of known protein sequences have human-verified functions. While deep learning methods have demonstrated promise for protein function prediction current models are limited to predicting only those functions on which they were trained. 

ProtNote is a multimodal deep learning model that leverages free-form text to enable both supervised and zero-shot protein function prediction. ProtNote not only maintains near state-of-the-art performance for annotations in its train set, but also generalizes to unseen and novel functions in zero-shot test settings.

- **Developed by:** Samir Char, Nathaniel Corley, Sarah Alamdari, Kevin K. Yang, Ava P. Amini
- **Shared by [optional]:** Microsoft Research New England
- **Model type:** ProtNote is two tower Deep Learning model for protein funciton prediction. The first tower is a Convolutional Neural Network that encodes protein sequences, while the second is a Transformer used to encode protein function text descriptions. These two representations are combined through and output Multi Layer Perceptron that computes the final predictions.
- **Language(s) (NLP):** English
- **License:** [MIT License](https://opensource.org/licenses/MIT)
- **Finetuned from model [optional]:** 
    -  https://huggingface.co/intfloat/multilingual-e5-large-instruct
    - https://github.com/google-research/proteinfer/tree/master

### Model Sources [optional]

<!-- Provide the basic links for the model. -->

- **Repository:** https://github.com/microsoft/protnote/tree/main
- **Paper [optional]:** TODO

## Uses


### Direct Use

This model is intended for research use. It can be used to predict the probability that the given protein sequences carry out a function described via free-from text. We provide model weights under five different seeds. Detailed use instructions are available in the model repo.


## Bias, Risks, and Limitations

- This model is intended for use on protein sequences. It is not meant for other biological sequences, such as DNA sequences.
- Even though the model could take any type of text as input, we recommend providing only functional descriptions. Non-functional descriptions may exhibit reduced performance.
- The model may be biased toward the content, vocabulary, and language style of the functional descriptions in the Gene Ontology, which is the dataset used to train the model.


## How to Get Started with the Model

Use the code below to install our model:

```
git clone https://github.com/microsoft/protnote.git
cd protnote
conda env create -f environment.yml
conda activate protnote
pip install -e ./  # make sure ./ is the dir including setup.py
```

For detailed instructions on package usage, please refer to the README in model repo: https://github.com/microsoft/protnote.git


## Training Details


### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

ProtNote is trained with protein sequences from the SwissProt section of the [UniProt](http://www.uniprot.org/) database, the world’s largest repository of manually curated, high-quality protein sequence and function information. 80% of the sequences are randomly assigned to train, 10% to validation, and 10% to test; we use the assignments reported by ProteInfer. We remove duplicated sequences and long sequences with more than 10,000 amino acids.

Further, the model is trained on descriptions from [Gene Ontology](https://geneontology.org/) (GO) annotations. GO annotations capture our knowledge of three aspects of protein biology: molecular function, biological process, and cellular component. Consequently, GO terms are phrases describing the molecular actions of gene products, the biological processes in which those actions occur, and the cellular locations where they are present. We train our model with the GO terms descriptions from the [01 July 2019 GO Annotations release](https://release.geneontology.org/2019-07-01/ontology/go.obo).



### Training Procedure

To build a model capable of both supervised and zero-shot
protein-function annotation, we formulate Gene Ontology (GO) term prediction from protein sequence as a binary classification problem: given a protein-GO term pair, the model predicts if the protein has the presented GO term. To combat class imbalance ProtNote leverages Focal Loss and weighted sampling of the sequences according to the inverse frequency of their functional annotations.

ProtNote uses ProteInfer as a protein sequence encoder and Multilingual E5 Text Embedding model as a text encoder – both encoders are frozen throughout training. To transform the embeddings to a more suitable space and reshape them to a common size d = 1024, they are each passed through independent multi-layer perceptrons (MLP) with three hidden layers of 3d = 3072 units and an output layer of size d = 1024. These embeddings are concatenated and fed through an MLP with three hidden layers of 3d = 3072 units, followed by the output neuron. The MLPs have no bias parameters because we use batch normalization and ReLU activations.

ProtNote is trained for 46 epochs on 8x 32GB V100 NVIDIA GPUs using an effective batch size of 256 (32 x 8) with dynamic padding. The model trains using Adam optimizer with learning rate of 0.0003 and employs gradient clipping set to 1 and mixed precision [26]. We select the model checkpoint based on best validation performance


## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data & Metrics

#### Testing Data

We evaluate ProtNote in both the supervised and zero-shot settings, benchmarking against the state-of-the-art deep-learning method ProteInfer and the gold-standard, homology-based method BLAST in the supervised setting and against custom embedding-based baselines in the zero-shot setting. For completeness, we break down the models’ performance across the three GO Ontologies (biological process, cellular component, molecular function) and the seven top-level EC classes (oxidoreductases, transferases hydrolases, lyases, isomerases, ligases, and translocases).

To asses ProtNote's ability to predict unseen/novel funcitonal text descriptions, we test it on unseen sequences *and* annotations from the July 2024 releases (data 5 years older than the training data) of the Gene Ontology ([Jul 2024 GO Release](https://release.geneontology.org/2024-06-17/ontology/go.obo)) and the Enzyme Commission (a data the model was never trained on) ([EC descriptions](https://ftp.expasy.org/databases/enzyme/enzclass.txt), [EC annotations](https://ftp.expasy.org/databases/enzyme/enzyme.dat)).


#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

To evaluate the performance of different models, we use both macro- and micro-averaged mean Average Precision (mAP), also known as the Area Under the Precision-Recall Curve (AUPRC). The mAP metric summarizes model performance across all possible prediction thresholds, eliminating threshold tuning and providing a more robust assessment of model performance than threshold-dependent metrics such as F1 or Fmax scores. We report both mAP Macro and mAP Micro because of their different virtues, although we use mAP Macro for model selection.

### Results

In this work, we introduce ProtNote, a multimodal deep learning model capable of supervised and zero-shot protein function annotation. We demonstrate how ProtNote leverages unstructured text to predict the likelihood that proteins perform arbitrary functions. Importantly, we showcase that ProtNote generalizes to both unseen sequences and functions by evaluating the model on newly-added GO annotations and on enzyme annotations, which were not used to train the model. We observe that this generality does not come at the cost of supervised performance; our model is also performant at predicting known GO terms, performing on par with the state-of-the-art model ProteInfer.

## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

- **Hardware Type:** 8 x 32GB NVIDIA V100 GPUs
- **Hours used:** 960 = 5 days x 24h x 8 GPUs
- **Cloud Provider:** Azure
- **Compute Region:** East US 2
- **Carbon Emitted:** 106.5 kg CO2

## Citation [optional]

TODO:

**BibTeX:**

{{ citation_bibtex | default("[More Information Needed]", true)}}

**APA:**

{{ citation_apa | default("[More Information Needed]", true)}}
