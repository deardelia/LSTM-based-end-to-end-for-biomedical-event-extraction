# LSTM-Based End-to-End Framework for Biomedical Event Extraction

(This work have been published in IEEE/ACM, TCBB, 2019)
https://www.ncbi.nlm.nih.gov/pubmed/31095491
## Overview

Biomedical event extraction plays an important role in the extraction of biological information from large-scale scientific
5 publications. However, most state-of-the-art systems separate this task into several steps, which leads to cascading errors. In addition, it is
6 complicated to generate features from syntactic and dependency analysis separately. Therefore, in this paper, we propose an end-to-end
7 model based on long short-term memory (LSTM) to optimize biomedical event extraction. Experimental results demonstrate that our
8 approach improves the performance of biomedical event extraction. We achieve average F1-scores of 59.68, 58.23, and 57.39 percent on
9 the BioNLP09, BioNLP11, and BioNLP13’s Genia event datasets, respectively. The experimental study has shown our proposed model’s
10 potential in biomedical event extraction

## Experiment Detials
### Dataset
In this study, two datasets were employed. The first is pre
pared for training Word2Vec and is composed of unlabeled
abstract texts from a public database called PubMed, which
is approximately 6 GB in size, including biological publica
tions from 2010 to 2017. These abstracts were prepared for
training Word2Vec because some rare biological terms
require more data to construct more accurate word embeddings. The other dataset is downloaded from BioNLP’09, BioNLP’11 and BioNLP’13, which consist of training sets, development sets, and test sets. 

### Word Embedding
The primary purpose of word embedding is to form a
lookup table to represent each word in a vocabulary with dense, low dimensional, and real-valued vectors
         

### Bi-LSTM-Based Trigger Detection
A Bi-LSTM network can access both preceding and
subsequent contexts. It aims to map every word in a dictionary to a numerical vector such that the distance and relationship between vectors reflects the semantic information
between words. Therefore, in this step, inputs are composed
of vectors corresponding to the words in a sentence.
### Tree-LSTM-Based Argument Detection
First, we create candidate pairs according to the trigger 
word found in previous step. Then we build a dependency 
tree layer by employing Tree-LSTM. According to, features of nodes are mainly constructed based on the shortest
dependency path. Next, for each candidate pair, its new
vector can be represented by the combination of its corresponding hidden layer in both Bi-LSTM network and Tree-LSTM network. The softmax layer then receives the target candidate vectors and will make a prediction

### End-to-End Model Training
In this research, we combine Tree-LSTM (corresponding to 
argument detection) and Bi-LSTM (corresponding to trigger 
detection), as the inputs of the argument detection are composed of vectors from the Bi-LSTM layer and Tree-LSTM
layer.
