# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 15:41:21 2022

@author: anonymous
"""

import numpy as np

import demo
import soat_utils as utils

encoding_type = demo.code_mode
model_name = demo.model_name
topic_num = demo.topic_num
# # # reading the generative results
# # prepare file names of results
# word-index dictionary file name
word2idx_filename_SOAT = demo.save_p + model_name + 'word_index.txt'
# index-word dictionary file name
idx2word_filename_SOAT = demo.save_p + model_name + 'index_word.txt'
# document-anchor words dictionary file name
docs_anchor_words_dict_filename_SOAT = demo.save_p + model_name + 'docs_anchor_words_dict' + str(topic_num) + '.txt'
# topic-word distribution file name
tw_dist_filename_SOAT = demo.save_p + model_name + 'topic_word_distribution_' + str(topic_num) + '.npy'
# document-topic distribution file name
dt_dist_filename_SOAT = demo.save_p + model_name + 'doc_topic_distributions_' + str(topic_num) + '.npy'
# class-specific topic-word distribution file name
lb_tw_dist_filename_SOAT = demo.save_p + model_name + 'labeled_topic_word_distribution_' + str(topic_num) + '.npy'
# file name of topic assignment for all words of each document
docs_filename_SOAT = demo.save_p + model_name + 'docs_list' + str(topic_num) + '.npy'

# # reading all results
# read the id-word and word-id dictionaries
f = open(idx2word_filename_SOAT, 'r', encoding=encoding_type)
data = f.read()
SOAT_idx2word = eval(data)
f.close()
f = open(word2idx_filename_SOAT, 'r', encoding=encoding_type)
data = f.read()
SOAT_word2idx = eval(data)
f.close()
# read the doc-anchor dictionary
f = open(docs_anchor_words_dict_filename_SOAT, 'r', encoding=encoding_type)
docs_anchor_words_dict = f.read()
docs_anchor_words_dict = docs_anchor_words_dict.replace("array", "np.array").replace("dtype=int64", "")
docs_anchor_words_dict = eval(docs_anchor_words_dict)
f.close()
# read the class-specific topic-word distributions
SOAT_labeled_topic_word_distributions = np.load(lb_tw_dist_filename_SOAT)
# read the topic-word distributions
SOAT_topic_word_distributions = np.load(tw_dist_filename_SOAT)
# read the doc-topic distributions
SOAT_doc_topic_distributions = np.load(dt_dist_filename_SOAT)
# read the topic assignment for all words of each document
SOAT_docs_list = np.load(docs_filename_SOAT, allow_pickle=True)

if __name__ == '__main__':
    # set a test word
    word = "key"

    topk_words = 100
    data = demo.test_data
    true_label_list = np.array(demo.plabels_ground)
    label_list = np.array(demo.valid_plabels)

    words_num = len(SOAT_idx2word)
    topic_num = demo.topic_num
    label_num = len(set(true_label_list))

    SOAT_labeled_topic_word_dist = SOAT_labeled_topic_word_distributions
    SOAT_doc_topic_dist = SOAT_doc_topic_distributions

    # compute top-k class-specific topical words
    SOAT_label_words = utils.get_label_words(SOAT_labeled_topic_word_dist[1:], SOAT_labeled_topic_word_dist[1:],
                                             SOAT_idx2word, topk_words)
    # output class-specific topical words
    print("class-specific topical words:")
    for i in range(label_num):
        topical_words = set(SOAT_label_words[i])
        print(topical_words)

    # compute top-k class-specific anchor words
    word_anchor_dict = utils.get_word_labeled_anchors(word,
                                                      SOAT_docs_list,
                                                      docs_anchor_words_dict,
                                                      true_label_list,
                                                      SOAT_word2idx,
                                                      SOAT_idx2word,
                                                      label_num, topk_words)
    # output class-specific anchor words for the given test word
    print("class-specific anchor words:")
    for label in range(1, label_num + 1):
        anchors = set([w[0] for w in word_anchor_dict[label]])
        print(anchors)
