# -*- coding: utf-8 -*-
"""
Created on Mon May 28 18:57:00 2022

@author: anonymous
"""
import warnings
warnings.filterwarnings("ignore")
import soat_utils as utils
import numpy as np
import soat_model as soat
import multiprocessing
from matplotlib import pyplot as plt

# # parameters setting
# give a name to the model
model_name = 'soat_'
# set the number of topic in modeling
topic_num = 20
# set the sampling rounds for parameter estimation
round_num = 5
# hyperparameter for doc-topic distribution
palpha = 0.05
# hyperparameter for topic-word distribution
pbeta = 0.05
# hyperparameter for class-specific topic distribution
pgamma = 0.05
# threshold for detecting anchor words
ptau = 0.35
# percentage of labeled documents (training data)
train_test_ratio = 0.8
# number of parallel processes, which does not exceed the number of cpu cores
proc_num = min(4, multiprocessing.cpu_count())
code_mode = "utf-8"
save_p = "test_result/"

# # dataset
data_name = '20Newsgroups'
data = np.load("demo-data/20Newsgroups_demo.npy", allow_pickle=True)
# label list for all documents
plabels_ground = np.load("demo-data/20Newsgroups_labels_demo.npy", allow_pickle=True)
# label list for only labeled documents, where unlabeled ones are denoted by 0, e.g., [1,2,0,0,...].
valid_plabels = utils.filter_labels(plabels_ground.copy(), train_test_ratio)
# dataset: a list of document and a document is a list of words, e.g., ([["a","b", "c"], ["d", "e"]]).
test_data = data


if __name__ == '__main__':
    print(test_data[:5])
    print(valid_plabels[:5])
    model = soat.SoatTopicModel(t_data=test_data,
                                plabel_list=valid_plabels,
                                plabels_ground=plabels_ground,
                                rounds=round_num,
                                save_p=save_p,
                                palpha=palpha,
                                pbeta=pbeta,
                                pgamma=pgamma,
                                ptau=ptau,
                                proc_num=proc_num,
                                topic_num=topic_num,
                                code_mode=code_mode,
                                model_name=model_name)
    multiprocessing.freeze_support()
    model.train()
    result_path = model.save_path
    y1 = np.load(str(result_path) + model_name + "per_list" + str(topic_num) + ".npy")
    x = np.linspace(0, round_num, round_num)
    plt.plot(x[::1], y1[:], "c*-", label='SOAT', linewidth=1)
    plt.title("Convergence Test By Perplexities")
    plt.ylabel(u"Perplexities")
    plt.xlabel(u"rounds")
    plt.legend(loc="upper right")
    plt.show()

