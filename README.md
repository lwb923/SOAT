# A Topic Model - SOAT

 A class-specific sense aware topic model based on soft orthogonalized topics for document classification.  
 We aim to model the class-specific word senses in topic space. The challenge is to optimize the class separability of the senses, i.e., obtaining sense vectors with both high intra-class and low inter-class similarities. We use soft orthogonalization for topics, i.e., reserving all topics while learning a group of class-specific weights. Besides, we detect highly class-specific words in each document and use them to guide sense estimation.  
  
   
## Usage
  
This project contains four files: "soat_model.py", "soat_anchors_test.py", "soat_utils.py" and "demo.py".   
* "demo.py" shows a simple example of how to use SOAT for training documents.  
	* We uploaded a demo dataset which sampled from 20Newsgroups (including 20 classes and 1000 documents) for demonstration. The dataset and its labels are load as folows:
```python 
data_name = '20Newsgroups'
data = np.load("20Newsgroups_demo.npy", allow_pickle=True)
# label list for all documents
plabels_ground = np.load("20Newsgroups_labels_demo.npy", allow_pickle=True)
# label list for only labeled documents, where unlabeled ones are denoted by 0, e.g., [1,2,0,0,...].
valid_plabels = utils.filter_labels(plabels_ground.copy(), train_test_ratio)
# dataset: a list of document and a document is a list of words, e.g., ([["a","b", "c"], ["d", "e"]]).
test_data = data
```

	*Then the mode can be initialized and trained by the following steps:  
	
```python
model = soat.SoatTopicModel(t_data=test_data,
                            plabel_list=valid_plabels,
                            plabels_ground=plabels_ground,
                            rounds=round_num,
                            save_p=save_p,
                            palpha=palpha,
                            pbeta=pbeta,
                            ptau=ptau,
                            proc_num=proc_num,
                            topic_num=topic_num,
                            code_mode=code_mode,
                            model_name=model_name)
model.train()
```
     
* "soat_model.py" is the core code for our model SOAT.  
	*It produces 12 results: 

| Filename                          | Description                                                                                                                                                                            |
|-----------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `index_word`                  | index-word dictionary                                                                                                                                                                  |
| `word_index`                  | word-index dictionary                                                                                                                                                                  |
| `doc_topic_distributions`     | list of topic distributions for all documents                                                                                                                                          |
| `docs_anchor_words_dict`          | a dictionary of anchor words corresponding to each document                                                                                                                            |
| `docs_list`                       | all document-related data, including the word list in each document, the topic corresponding to the word, the dependence strength of the word on the anchor word,and the tf-idf weight |
| `labeled_topic_word_distribution` | class-specific topic-word distributions, i.e. class-specific list of word sense vectors                                                                                                |
| `labeled_topic_word`              | class-specific topic-word frequency lists                                                                                                                                              |
| `per_list`                        | list of perplexities per training rouds                                                                                                                                                |
| `pre_label_probs_list`            | the class prediction probabilities for each word in the document                                                                                                                       |
| `topic_word_distribution`         | topic-word distributions for the dataset                                                                                                                                               |
| `topic_word`                      | topic-word frequencies for the dataset                                                                                                                                                 |
| `word_scores_dict`                | a dictionary of the class-specific scores for all words                                                                                                                                |

   
* "soat_anchors_test.py" is an example for showing the class-specific topical words and class anchor words.  
	*We can set an example word (e.g., "key") as follow:  
```python  
# set a test word  
word = "key"  
``` 
* "soat_utils.py" is a file of common methods.  
  
  
**NOTE: 
1. We show the average class prediction accuracy of all words in the training steps.
2. We use and record the perplexities in each step of training to measure the convergence of model.  
3. In order to combine more flexibly with the model based on neural network, we keep the topic assignment of all words, so it will spend a bit more time in training. For this demo, it takes about several minutes. 
4. The topic assignments for words of each document can be obtained by the following steps:
```python 
import numpy as np
import demo

topic_num = demo.topic_num
model_name = demo.model_name
# file name of topic assignment for all words of each document
docs_filename_SOAT = demo.save_p + model_name + 'docs_list' + str(topic_num) + '.npy'
# read the topic assignment for all words of each document
SOAT_docs_list = np.load(docs_filename_SOAT, allow_pickle=True)

# define and obtain the topic assignments for words of each document 
topic_assignments_list = []
for doc in SOAT_docs_list:
    topic_assignments_list.append(doc[:, 1])
```  
 
	
