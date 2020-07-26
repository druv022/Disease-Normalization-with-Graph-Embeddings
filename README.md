# Disease-Normalization-with-Graph-Embeddings

## Table of Content:
1. Basic training and Testing
2. Structure of code
3. Paper

### Basic Usage:
    Please view the usage in train_NER_script.py, train_EL.py, train_MTL_script.py and test_NER.py, test_EL.py, test_MTL.py respectively for training and testing.

    Paths and Params are two object which needs to be initialized with appropriate values.

### Structure of Code:
Folder descriptions:
* **analysis**: Execute *analyze_data2.py* and *analyze_mesh.py* after updating the data paths.
* **config**: definition of Paths and Params objects
* **EL**: files containing the EL training and testing methods. *EL_partA.py* is used to train node2vec type-I and type-II. *EL_GCN.py* is used to train GCN. *EL_utils.py* contains functions used in common for EL. Rest are experimental files.
* **models**: definition of NER and EL models. Other than *summery.py*, the files define models specific to the task. *summery.py* is experimental.
* **MTL**: file containing training and test functions for Multitask Learning
* **NER**: file containing training and test functions for Named Entity Recognition using NERDS
* **nerds**: NERDS (contains updated NER models)
* **node2vec**: Contains files of basic *node2vec* algorithm (*main.py* and *node2vec.py* ) as well as unsupervised training files of the two variant Type-I (*node2vec1.py*) and Type-II (node2vec2.py). *SkipGram.py* and *Another_sk.py* are experimental files.
*  **process**: utility functions for preprocessing MeSH data
*  **utils**: Utility functions (with extra/old codes)
*  **wvlib_master**: Package used to read pre-trained PubMed word embedding.

### Paper:
Please cite:
*  Dhruba Pujary, Camilo Thorne, Wilker Aziz - *Disease Normalization with Graph Embeddings*, 2020, Proceedings of the IntelliSys 2020 conference (preprint: http://camilothorne.altervista.org/perso/intellisys-2020.pdf).
