# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 14:44:26 2022

@author: anonymous
"""
import math
import multiprocessing
import os
import time
import numpy as np
import soat_utils as utils


class SoatTopicModel:
    """
    Function description: A class-specific sense aware topic model based on soft orthogonalized topics. This model
    captures class-specific senses in a semi-supervised manner by a limited number of document class labels. Specifically
    , it keeps all topics for each word sense vector while learning a group of weights that correspond to its topics. A
    weight corresponds to the degree of importance of a topic in a sense vector to the class separability. These weights
    are further optimized with the guidance of the class anchor words in each document. Note that, a function named starting
    with "p_" is a parallel version of the original one.

    It produces 12 results: (1) index_word: index-word dictionary;
                            (2) word_index: word-index dictionary;
                            (3) doc_topic_distributions.npy: list of topic distributions for all documents;
                            (4) docs_anchor_words_dict: a dictionary of anchor words corresponding to each document;
                            (5) docs_list: all document-related data, including the word list in each document, the
                            topic corresponding to the word, the dependence strength of the word on the anchor word,
                            and the tf-idf weight (optional);
                            (6) labeled_topic_word_distribution: class-specific topic-word distributions, i.e. class-
                            specific list of word sense vectors.
                            (7) labeled_topic_word: class-specific topic-word frequency lists;
                            (8) per_list: list of perplexities per training rouds;
                            (9) pre_label_probs_list: the class prediction probabilities for each word in the document;
                            (10) topic_word_distribution: topic-word distributions for the dataset;
                            (11) topic_word: topic-word frequencies for the dataset;
                            (12) word_scores_dict: a dictionary of the class-specific scores for all words.

    Main Interface: (1) __init__(): initialization parameters;
                    (2) initialize(): parameter settings and initializations, including the number of documnets,topics, words
                     and other hyper paramters for dirichlet distribution;
                    (3) initial_pre_label_probs_list(): randomly initialize the label prediction probabilities for words in
                    each document.
                    (4) compute_labeled_topic_word(): calculate category-specific topic-word frequencies;
                    (5) create_dictionary_s(): create dictionary for dataset;
                    (6) computer_words_anchor_scores(): compute the scores of anchor words;
                    (7) compute_docs_anchor_words_dict(): calculate the anchor word corresponding to each document;
                    (8) get_a_topic(): assign topics to a word according to the topic distribution;
                    (9) initialize_distributions(): initialize all probability distributions;
                    (10) initial_docs_list(): initialize data such as documents, words, topic assignments, etc.
                    (11) initialize_tfidf(): initialize tf-idf weight parameters (optional)
                    (12) get_d_w_etas(): computes how dependent of words in a document on anchor words;
                    (13) compute_pred_labels(): compute label predictions for words in a document;
                    (14) compute_doc_topic(): computing the topic-assignment frequency of words in a document;
                    (15) compute_topic_word(): computing the topic-assignment frequency of all words;
                    (16) get_n_d_topics(): get the number of word w assigned by topic k in document d;
                    (17) get_labeled_n_w_topics(): get the number of word w assigned by any one topic k in all documents with a given label
                    (18) get_labeled_total_n_topics(): get the number of all words assigned by any one topic k with a given label;
                    (19) get_labeled_anchors_k(): calculate the frequency of a given topic based on anchor words;
                    (20) get_topic_sel_probs(): calculate class-specific weights for topics;
                    (21) get_labeled_n_ws_topics(): get the number of word w assigned by topic k in all documents except the current one;
                    (22) recompute_d_topic_distribution(): computes the topic distribution for a given document;
                    (23) gibbs_sampling(): gibbs sampling main procedure;
                    (24) compute_perplexities(): compute the perplexity of the training dataset;
                    (25) parameter_estimation(): parameter estimation main procedure;
                    (26) save_result(): save the results of topic modeling;
                    (27) train(): main procedure of the training for topic modeling.
    Modification record:
    """

    def __init__(self,
                 t_data,
                 plabel_list,
                 plabels_ground,
                 rounds=10,
                 save_p=None,
                 ptfidt_weights=None,
                 palpha=0.05,
                 pbeta=0.05,
                 pgamma=0.05,
                 ptau=0.35,
                 pdocs_list_tLDA=None,
                 proc_num=1,
                 topic_num=10,
                 code_mode="utf-8",
                 model_name="m_LGST"
                 ):
        """
            Function description: initialize parameters of model;
            Parameters: (1) t_data: training data, which is a list of document and a document is a list of words.
                        ([["a","b", "c"], ["d", "e"]])
                        (2) plabel_list: class labels of labeled documents, i.e., a list of
                        labels where 0 refers to unlabeled document ([1,0,1,2,1,1...]);
                        (3) plabels_ground: class labels of both labeled and unlabeled documents;
                        (4) rounds: the rounds number of sampling for the whole dataset;
                        (5) save_p: the path of saving results;
                        (6) ptfidt_weights: tf-idf weights for each word in each document ([[0.5,0.7],[0.1,0.8],...]);
                        (7) palpha: hyper-parameter for document-topic distribution;
                        (8) pbeta: hyper_parameter for topic-word distribution;
                        (9) pgamma: hyper_parameter for class-specific topic distribution;
                        (10) ptau: threshold for class anchor word score;
                        (11) pdocs_list_tLDA: pretrained topic model (optional);
                        (12) proc_num: number of parallel processes;
                        (13) topic_num: number of topics;
                        (14) code_mode: encoding of text;
                        (15) model_name: name of model instance.
            Return:
            Modification Record:
        """
        self.tfidf_weights = []
        self.pre_label_list = []
        self.pre_label_probs_list = []
        self.docs_list = []
        self.label_topic_distributions = []
        self.doc_topic_distributions = []
        self.topic_word_distribution = []
        self.perplexities = []
        self.per_list = []
        self.labeled_topic_word_distributions = []
        self.word_index = dict()
        self.index_word = dict()
        self.label_list_blocks = dict()
        self.pre_label_list_blocks = dict()
        self.labeled_topic_weights = dict()
        self.pre_label_probs_list_blocks = dict()
        self.word_scores_dict = dict()
        self.docs_anchor_words_dict = dict()
        self.docs_anchor_words_dict_blocks = dict()
        self.docs_list_blocks = dict()
        self.save_path_tLDA = ""
        self.dt_dist_filename_tLDA = ""
        self.tLDA_doc_topic_distributions = ""
        self.topic_word = 0 * np.ones([1, 1])
        self.doc_topic = 0 * np.ones([1, 1])
        self.labeled_topic_word = 0 * np.ones([1, 1, 1])

        if save_p is None:
            time_str = time.strftime("%a%b%d%H%M%S%Y", time.localtime())
            folder = os.getcwd() + "/" + time_str + "/"
            is_exist = os.path.exists(folder)
            if not is_exist:
                os.makedirs(folder)
            else:
                pass
            self.save_path = folder
        else:
            self.save_path = save_p
            pass
        self.p_num = proc_num
        self.code_mode = code_mode
        self.model_name = model_name
        self.p_tfidt_weights = ptfidt_weights
        self.p_docs_list = pdocs_list_tLDA
        self.data = t_data
        self.create_dictionary_s(self.data, self.save_path)
        self.round_num = rounds
        self.words_num = len(self.word_index)
        self.docs_num = len(self.data)
        self.topic_num = topic_num
        self.label_num = len(set(plabel_list))
        self.block_len = int(len(self.data) / self.p_num)
        self.alpha = palpha
        self.beta = pbeta
        self.gamma = pgamma
        self.tau = ptau
        self.labels_true = plabels_ground
        self.label_list = np.array(plabel_list)
        self.initialize()

    def initialize(self):
        """
            Function description: parameter settings and initializations, including the number of documnets,
            topics, words and other hyper paramters for dirichlet distribution
            Parameters:
            Return:
            Modification Record:
        """
        print("initializing...")
        self.topic_word = 0 * np.ones([self.topic_num, self.words_num])
        self.doc_topic = 0 * np.ones([self.docs_num, self.topic_num])
        self.labeled_topic_word = 0 * np.ones([self.label_num, self.topic_num, self.words_num], dtype=np.float32)
        self.initial_docs_list()
        self.initial_pre_label_probs_list()
        self.initialize_tfidf(weights=self.p_tfidt_weights)
        self.initialize_distributions()
        self.initialize_values_docs_list()
        self.initial_pre_label_list()
        self.p_initial_docs_list()
        self.p_initialize_values_labels_list()
        self.p_initialize_values_docs_list()
        self.compute_doc_topic()
        self.compute_topic_word()
        self.compute_labeled_topic_word()
        self.recompute_distributions()
        self.computer_words_anchor_scores()
        self.compute_docs_anchor_words_dict(self.tau)
        self.init_pred_labels()
        self.p_init_pred_labels()
        print("initialization finished")
        return

    def initial_pre_label_probs_list(self):
        """
            Function description: randomly initialize the label prediction probabilities for words in each document.
            Parameters:
            Return:
            Modification Record:
        """
        self.pre_label_probs_list = list(self.pre_label_probs_list)
        self.pre_label_probs_list.clear()
        for i in range(len(self.label_list)):
            if self.label_list[i] != 0:
                temp_probs = np.zeros([self.label_num - 1])
                temp_probs[self.label_list[i] - 1] = 1
                self.pre_label_probs_list.append(temp_probs)
            else:
                temp_probs = 0.5 * np.ones([self.label_num - 1])
                self.pre_label_probs_list.append(temp_probs)
        return

    def compute_labeled_topic_word(self):
        """
            Function description: calculate category-specific topic-word frequencies.
            Parameters:
            Return:
            Modification Record:
        """
        self.labeled_topic_word = np.array(self.labeled_topic_word)
        self.labeled_topic_word = 0 * self.labeled_topic_word
        for i in range(len(self.docs_list)):
            for j in range(0, len(self.docs_list[i])):
                self.labeled_topic_word[self.pre_label_list[i]][int(self.docs_list[i][j][1])][
                    int(self.docs_list[i][j][0])] += 1 * self.docs_list[i][j][2]
        self.labeled_topic_word[0] = self.topic_word
        return

    def create_dictionary_s(self, data, path):
        """
            Function description: create dictionary for dataset.
            Parameters:(1) data: training dataset;
                       (2) path: saving path for the dictionary
            Return:
            Modification Record:
        """
        if not os.path.exists(path):
            os.makedirs(path)
        for doc in data:
            for w in doc:
                if w not in self.word_index:
                    self.word_index[w] = len(self.word_index)
        self.index_word = dict(zip(self.word_index.values(), self.word_index.keys()))
        f = open(path + str(self.model_name) + 'word_index.txt', 'w', encoding=self.code_mode)
        f.write(str(self.word_index))
        f.close()
        f = open(path + str(self.model_name) + 'index_word.txt', 'w', encoding=self.code_mode)
        f.write(str(self.index_word))
        f.close()

    def computer_words_anchor_scores(self):
        """
            Function description: compute the scores of anchor words.
            Parameters:
            Return:
            Modification Record:
        """
        self.label_topic_distributions = self.labeled_topic_word[:].sum(axis=2) + self.beta
        self.label_topic_distributions = self.label_topic_distributions / self.label_topic_distributions.sum(
            axis=1).reshape(len(self.label_topic_distributions), 1)
        self.labeled_topic_word_distributions = np.array(self.labeled_topic_word_distributions)
        self.label_topic_distributions = np.array(self.label_topic_distributions)
        label_probs = dict()
        for label in range(1, self.label_num):
            label_probs[label] = list(self.label_list).count(label) / len(self.label_list)

        for d in range(len(self.docs_list)):
            d_dist = self.doc_topic_distributions[d]
            if self.label_list[d] == self.label_list[d]:
                is_pos = 1
            else:
                is_pos = -1
            for w in range(len(self.docs_list[d])):
                score_w = 0
                if self.label_list[d] == 0 or w not in self.docs_list[d][:, 0]:
                    continue
                else:
                    for label in range(1, self.label_num):
                        w_label_dist = self.labeled_topic_word_distributions[label, :, w]
                        label_dist = self.label_topic_distributions[label, :]
                        score_w += is_pos * label_probs[label] * (label_dist * w_label_dist).sum()
                if self.docs_list[d][w][0] in self.word_scores_dict.keys():
                    self.word_scores_dict[int(self.docs_list[d][w][0])] += score_w
                else:
                    self.word_scores_dict[int(self.docs_list[d][w][0])] = score_w

        return

    def compute_docs_anchor_words_dict(self, sub_threshold):
        """
            Function description: calculate the anchor word corresponding to each document.
            Parameters:(1) sub_threshold: threshold of detecting class anchor word, i.e., tau.
            Return:
            Modification Record:
        """
        self.docs_anchor_words_dict.clear()
        for d in range(len(self.docs_list)):
            candidates = []
            for w in range(len(self.docs_list[d])):
                if int(self.docs_list[d][w][0]) in self.word_scores_dict.keys():
                    candidates.append(self.word_scores_dict[int(self.docs_list[d][w][0])])
            idxs = np.argwhere(np.array(candidates) > sub_threshold)

            if len(idxs) == 0:
                idxs = list(range(len(self.docs_list[d])))
                np.random.shuffle(idxs)
                self.docs_anchor_words_dict[d] = idxs[0:]
            else:
                self.docs_anchor_words_dict[d] = idxs.squeeze(axis=1)
        return

    @staticmethod
    def get_a_topic(doc_topic_distribution):
        """
            Function description: assign topics to a word according to the topic distribution.
            Parameters:(1) doc_topic_distribution: topic distributions for a given document.
            Return: (1) topic assignment list for each word of the document
            Modification Record:
        """
        topics = np.random.multinomial(1, doc_topic_distribution)
        topic = [i for i, e in enumerate(topics) if e > 0][0]
        return topic

    def initialize_distributions(self):
        """
            Function description: initialize all probability distributions.
            Parameters:
            Return:
            Modification Record:
        """
        self.doc_topic_distributions = list(self.doc_topic_distributions)
        self.topic_word_distribution = list(self.topic_word_distribution)
        self.labeled_topic_word_distributions = list(self.labeled_topic_word_distributions)
        self.doc_topic_distributions.clear()
        self.topic_word_distribution.clear()
        self.labeled_topic_word_distributions.clear()
        for i in range(0, self.docs_num):
            self.doc_topic_distributions.append(1. / self.topic_num * np.ones([self.topic_num]))
        for i in range(0, self.topic_num):
            self.topic_word_distribution.append(1. / self.words_num * np.ones([self.words_num]))
        for label in range(self.label_num):
            self.labeled_topic_word_distributions.append(self.topic_word_distribution.copy())
        return

    def initial_docs_list(self):
        """
            Function description: initialize data such as documents, words, topic assignments, etc.
            Parameters:
            Return:
            Modification Record:
        """
        self.docs_list = list(self.docs_list)
        self.docs_list.clear()
        for doc in self.data:
            self.docs_list.append(np.ones([len(doc), 4], dtype=np.float))
        return

    def initialize_values_docs_list(self):
        """
            Function description: random setting values for docs_list.
            Parameters:
            Return:
            Modification Record:
        """
        if self.p_docs_list is None:
            for d in range(0, len(self.data)):
                for w in range(0, len(self.data[d])):
                    self.docs_list[d][w] = np.array([self.word_index[self.data[d][w]],
                                                     self.get_a_topic(self.doc_topic_distributions[d]),
                                                     self.tfidf_weights[d][w],
                                                     self.label_list[d]])
        else:
            for d in range(0, len(self.p_docs_list)):
                for w in range(0, len(self.p_docs_list[d])):
                    self.docs_list[d][w] = np.array([self.p_docs_list[d][w][0],
                                                     self.p_docs_list[d][w][1],
                                                     self.p_docs_list[d][w][2],
                                                     self.label_list[d]])
        return

    def p_initialize_values_labels_list(self):
        """
            Function description: random setting values for docs_list.
            Parameters:
            Return:
            Modification Record:
        """
        block_len = int(len(self.label_list) / self.p_num)
        for i in range(0, self.p_num):
            self.label_list_blocks[i] = self.label_list[i * block_len: i * block_len + block_len]
        return

    def initialize_tfidf(self, weights=None):
        """
            Function description: initialize tf-idf weight parameters (optional).
            Parameters: (1) weights: tf-idf weights for each word in each document ([[0.5,0.7],[0.1,0.8],...]);
            Return:
            Modification Record:
        """
        self.tfidf_weights = list(self.tfidf_weights)
        self.tfidf_weights.clear()
        if weights is None:
            for doc in self.data:
                self.tfidf_weights.append(np.ones([len(doc)], dtype=np.uint8))
        else:
            self.tfidf_weights = weights
        return

    def p_initial_docs_list(self):
        self.docs_list_blocks.clear()
        block_len = int(len(self.data) / self.p_num)
        for i in range(0, self.p_num):
            data_block = self.data[i * block_len: i * block_len + block_len]
            sub_docs_list = []
            for doc in data_block:
                sub_docs_list.append(np.ones([len(doc), 4], dtype=np.float))
            self.docs_list_blocks[i] = sub_docs_list
        return

    def p_initialize_values_docs_list(self):
        block_len = int(len(self.data) / self.p_num)
        for i in range(0, self.p_num):
            data_block = self.data[i * block_len: i * block_len + block_len]
            for d in range(0, len(data_block)):
                for w in range(0, len(data_block[d])):
                    self.docs_list_blocks[i][d][w] = self.docs_list[i * block_len + d][w]
        return

    def initial_pre_label_list(self):
        self.pre_label_list = list(self.pre_label_list)
        self.pre_label_list.clear()
        for i in range(len(self.label_list)):
            self.pre_label_list.append(self.label_list[i])
        return

    @staticmethod
    def get_d_w_etas(labeled_topics_ws,
                     labeled_total_topics,
                     labeled_anchors_topics,
                     labeled_anchors_total_topics,
                     sub_beta=0.05):
        """
            Function description: computes how dependent of words in a document on anchor words;
            Parameters:(1) labeled_topics_ws: numbers of a given word for each topic and each label;
                       (2) labeled_total_topics: numbers of all words for each topic and each label;
                       (3) labeled_anchors_total_topics: numbers of anchor words for each topic and each label;
                       (4) sub_beta: parameter for class-specific topic word distributions.
            Return: (1) d_ws_etas: dependencies for each word in a document.
            Modification Record:
        """
        sub_label_num = len(labeled_topics_ws[1:])
        sub_topic_num = len(labeled_topics_ws[1, :, 0])
        w_probs = (sub_beta + labeled_topics_ws[1:, :, :]) / \
                  (sub_beta * sub_label_num + labeled_total_topics[1:, :].reshape(sub_label_num, sub_topic_num, 1))
        anchors_probs = (sub_beta + labeled_anchors_topics[1:, :]) / (
                sub_beta * sub_label_num + labeled_anchors_total_topics[1:, :])
        sum_probs = w_probs + anchors_probs.reshape(sub_label_num, sub_topic_num, 1)
        d_ws_etas = anchors_probs.reshape(sub_label_num, sub_topic_num, 1) / sum_probs
        return d_ws_etas

    def p_compute_pred_labels(self):
        """
            Function description: compute label predictions for words in a document.
            Parameters:
            Return:
            Modification Record:
        """
        self.doc_topic_distributions = np.array(self.doc_topic_distributions)
        self.topic_word_distribution = np.array(self.topic_word_distribution)
        self.labeled_topic_word_distributions = np.array(self.labeled_topic_word_distributions)
        for i in range(len(self.docs_list)):
            if self.label_list[i] != 0:
                self.pre_label_list[i] = self.label_list[i]
                probs = np.zeros(self.label_num - 1)
                probs[self.label_list[i] - 1] = 1
                self.pre_label_probs_list[i] = probs
                continue
            else:
                pass
            w_idxs = [int(item) for item in self.docs_list[i][:, 0]]
            l_tw_dist = self.labeled_topic_word_distributions[1:, :, w_idxs]
            dt_dist = 1e1 * self.doc_topic_distributions[i, :].reshape(1, self.topic_num, 1)
            likelihood = (dt_dist * l_tw_dist + 1.7e-308).sum(1).prod(1) + 1.7e-308
            # label_keys = list(range(1, self.label_num))
            # freq_labels_dict = Counter(self.pre_label_list)
            # freq_labels = list(map(freq_labels_dict.get, label_keys))
            likelihood_list = likelihood
            likelihood_dist = np.array(likelihood_list) / np.sum(likelihood_list)
            pred_labels = np.random.multinomial(1, likelihood_dist)
            pred_label = [i for i, e in enumerate(pred_labels) if e > 0][0]
            self.pre_label_list[i] = pred_label + 1
            self.pre_label_probs_list[i] = likelihood_dist
        self.p_init_pred_labels()
        return

    def init_pred_labels(self):
        """
            Function description: initialize word label predictions.
            Parameters:
            Return:
            Modification Record:
        """
        self.doc_topic_distributions = np.array(self.doc_topic_distributions)
        self.topic_word_distribution = np.array(self.topic_word_distribution)
        self.labeled_topic_word_distributions = np.array(self.labeled_topic_word_distributions)
        for i in range(len(self.docs_list)):
            if self.label_list[i] != 0:
                self.pre_label_list[i] = self.label_list[i]
                probs = np.zeros(self.label_num - 1)
                probs[self.label_list[i] - 1] = 1
                self.pre_label_probs_list[i] = probs
                continue
            else:
                pass
            w_idxs = [int(item) for item in self.docs_list[i][:, 0]]
            l_tw_dist = self.labeled_topic_word_distributions[1:, :, w_idxs]
            dt_dist = 1e1 * self.doc_topic_distributions[i, :].reshape(1, self.topic_num, 1)
            likelihood = (dt_dist * l_tw_dist + 1.7e-308).sum(1).prod(1) + 1.7e-308
            # label_keys = list(range(1, self.label_num))
            # freq_labels_dict = Counter(self.pre_label_list)
            # freq_labels = list(map(freq_labels_dict.get, label_keys))
            likelihood_list = likelihood
            likelihood_dist = np.array(likelihood_list) / np.sum(likelihood_list)
            pred_labels = np.random.multinomial(1, likelihood_dist)
            pred_label = [i for i, e in enumerate(pred_labels) if e > 0][0]
            self.pre_label_list[i] = pred_label + 1
            self.pre_label_probs_list[i] = likelihood_dist
        return

    def refresh_docs_labels(self):
        """
            Function description: update labels for documents in each iteration.
            Parameters:
            Return:
            Modification Record:
        """
        for d in range(len(self.docs_list)):
            self.docs_list[d][:, 3] = self.pre_label_list[d] * np.ones(len(self.docs_list[d]))

    def p_init_pred_labels(self):
        block_len = int(len(self.pre_label_list) / self.p_num)
        for i in range(0, self.p_num):
            if i in self.pre_label_list_blocks:
                self.pre_label_list_blocks[i].clear()
                self.pre_label_probs_list_blocks[i].clear()
            else:
                pass
            self.pre_label_list_blocks[i] = self.pre_label_list[i * block_len: i * block_len + block_len]
            self.pre_label_probs_list_blocks[i] = self.pre_label_probs_list[i * block_len: i * block_len + block_len]
        return

    def compute_doc_topic(self):
        """
            Function description: computing the topic-assignment frequency of words in a document.
            Parameters:
            Return:
            Modification Record:
        """
        self.doc_topic = np.array(self.doc_topic)
        self.doc_topic = 0 * self.doc_topic
        for i in range(len(self.docs_list)):
            for j in range(0, len(self.docs_list[i])):
                self.doc_topic[i][int(self.docs_list[i][j][1])] += 1 * self.docs_list[i][j][2]

    def compute_topic_word(self):
        """
            Function description: computing the topic-assignment frequency of all words.
            Parameters:
            Return:
            Modification Record:
        """
        self.topic_word = np.array(self.topic_word)
        self.topic_word = 0 * self.topic_word
        for i in range(len(self.docs_list)):
            for j in range(0, len(self.docs_list[i])):
                self.topic_word[int(self.docs_list[i][j][1])][int(self.docs_list[i][j][0])] += 1 * self.docs_list[i][j][
                    2]
        return

    @staticmethod
    def p_get_n_d_topics(d, block, sub_docs_list_blocks, t_len):
        """
            Function description: get the number of word w assigned by topic k in document d.
            Parameters: (1) d: document id;
                        (2) block: id of the parallelized data block;
                        (3) sub_docs_list_blocks: parallelized text data;
                        (4) t_len: topic number.
            Return:
            Modification Record:
        """
        idxs_topics = [sub_docs_list_blocks[block][d][:, 1] == k for k in range(t_len)]
        n_d_topics = np.array([sub_docs_list_blocks[block][d][:, 2][idxs].sum() for idxs in idxs_topics])
        return n_d_topics

    @staticmethod
    def p_get_labeled_n_w_topics(d, w, block, sub_docs_list_blocks, sub_labeled_topic_word):
        """
            Function description: get the number of word w assigned by any one topic k in all documents with a given label.
            Parameters: (1) d: document id;
                        (2) w: word id;
                        (3) block: id of the parallelized data block;
                        (4) sub_docs_list_blocks: parallelized text data;
                        (5) sub_labeled_topic_word: class-specific topic-word distribution.
            Return: (1) labeled_n_w_topics: class-specific word numbers for each topic.
            Modification Record:
        """
        labeled_n_w_topics = sub_labeled_topic_word[:, :, int(sub_docs_list_blocks[block][d][w][0])]
        labeled_n_w_topics[int(sub_docs_list_blocks[block][d][w][3]),
                           int(sub_docs_list_blocks[block][d][w][1])] -= 1
        return labeled_n_w_topics

    @staticmethod
    def p_get_labeled_total_n_topics(sub_labeled_topic_word):
        """
            Function description: get the number of all words assigned by any one topic k with a given label.
            Parameters: (1) sub_labeled_topic_word: class-specific topic-word distribution.
            Return:
            Modification Record:
        """
        labeled_total_n_topics = sub_labeled_topic_word[:, :, :].sum(axis=2)
        return labeled_total_n_topics

    def p_get_labeled_anchors_k(self, anchors_idxs, sub_labeled_topic_word):
        """
            Function description: calculate the frequency of a given topic based on anchor words.
            Parameters: (1) anchors_idxs: ids of anchor words;
                        (2) sub_labeled_topic_word: class-specific topic-word distribution.
            Return:
            Modification Record:
        """
        labeled_n_topics = sub_labeled_topic_word[:, :, anchors_idxs].mean(2)
        labeled_n_totals = sub_labeled_topic_word.sum(2)
        return labeled_n_topics + self.beta, labeled_n_totals + self.beta

    @staticmethod
    def get_topic_sel_probs(labeled_topic_prob):
        """
            Function description: calculate class-specific weights for topics.
            Parameters: (1) labeled_topic_prob: class-specific probabilities for topics;
            Return: (1) topic_probs: weights for each topic on classes.
            Modification Record:
        """
        topic_labeled_probs = labeled_topic_prob.T
        topic_labeled_entropies = np.array([1 / utils.entropy(item) for item in topic_labeled_probs])
        topic_labeled_entropies = (
                                          topic_labeled_entropies - topic_labeled_entropies.min()) / topic_labeled_entropies.var()
        topic_probs = topic_labeled_entropies / topic_labeled_entropies.sum()
        return topic_probs

    @staticmethod
    def p_get_topic_sel_probs(labeled_topic_probs):  # [label, topic, word]
        """
            Function description: calculate class-specific weights for topics.
            Parameters: (1) labeled_topic_prob: class-specific probabilities for topics;
            Return: (1) topic_probs: weights for each topic on classes.
            Modification Record:
        """
        t_num = len(labeled_topic_probs[0, :, 0])
        w_num = len(labeled_topic_probs[0, 0, :])
        topic_labeled_ws_probs = np.array([labeled_topic_probs[:, :, i].T for i in range(w_num)])
        topic_labeled_entropies = np.array([1 / utils.entropy(item) for topic_labeled_probs in topic_labeled_ws_probs \
                                            for item in topic_labeled_probs])
        topic_labeled_entropies = topic_labeled_entropies.reshape(w_num, t_num)
        topic_labeled_entropies = (topic_labeled_entropies - topic_labeled_entropies.min(axis=1).reshape(w_num, 1)) \
                                  / topic_labeled_entropies.var(axis=1).reshape(w_num, 1)
        topic_labeled_entropies += 1.7e-308
        topic_ws_probs = topic_labeled_entropies / topic_labeled_entropies.sum(axis=1).reshape(w_num, 1)

        return topic_ws_probs

    @staticmethod
    def p_get_labeled_n_ws_topics(d, block, sub_docs_list_blocks, sub_labeled_topic_word):
        """
            Function description: get the number of word w assigned by topic k in all documents except the current one.
            Parameters: (1) d: document id;
                        (2) block: id of the parallelized data block;
                        (3) sub_docs_list_blocks: parallelized text data;
                        (4) sub_labeled_topic_word: class-specific topic-word distribution.
            Return: (1) labeled_n_w_topics: class-specific word numbers for each topic.
            Modification Record:
        """
        w_idxs = [int(idx) for idx in sub_docs_list_blocks[block][d][:, 0]]
        minus_poses = np.array([(int(sub_docs_list_blocks[block][d][i][3]),
                                 int(sub_docs_list_blocks[block][d][i][1]),
                                 i) for i in range(len(w_idxs))])
        labeled_n_ws_topics = sub_labeled_topic_word[:, :, w_idxs]
        labeled_n_ws_topics[minus_poses[:, 0], minus_poses[:, 1], minus_poses[:, 2]] -= 1
        return labeled_n_ws_topics

    def p_recompute_d_topic_distribution(self,
                                         sub_label_topic_distributions,
                                         sub_eta_labels,
                                         sub_pred_label_probs,
                                         n_d_topics,
                                         labeled_n_ws_topics,
                                         labeled_total_n_topics,
                                         labeled_n_anchors_topics,
                                         labeled_n_anchors_totals_topics,
                                         sub_words_num,
                                         sub_topic_num,
                                         sub_label_num,
                                         sub_doc_len,
                                         sub_alpha,
                                         sub_beta):
        """
            Function description: computes the topic distribution for a given document.
            Parameters: (1) sub_label_topic_distributions: class-specific topic distribution;
                        (2) sub_eta_labels: class-specific dependencies for each word on anchor words;
                        (3) sub_pred_label_probs: label prediction probabilities;
                        (4) n_d_topics: number of words in document d with each topic;
                        (5) labeled_n_ws_topics: class-specific word numbers for each topic;
                        (6) labeled_total_n_topics: number of all words assigned by any one topic k with a given label;
                        (7) labeled_n_anchors_topics: class-specific anchor word numbers for each topic;
                        (8) labeled_n_anchors_totals_topics: class-specific word numbers for each topic;
                        (9) sub_words_num: word numbers in total;
                        (10) sub_topic_num: topic numbers;
                        (11) sub_label_num: label numbers;
                        (12) sub_doc_len: length of the given document;
                        (13) sub_alpha: hyper-parameter of document-topic distribution;
                        (14) sub_beta: hyper-parameter of topic-word distribution.
            Return: (1) new_topic_distributions: topic distribution for the given document.
            Modification Record:
        """
        sub_pred_label_probs = sub_pred_label_probs.reshape(sub_label_num - 1, 1, 1)
        n_d_topics_label = np.ones([sub_label_num - 1, sub_topic_num, sub_doc_len])
        hybrid_ws_topics = np.zeros([sub_label_num - 1, sub_topic_num, sub_doc_len])
        hybrid_total_topics = np.zeros([sub_label_num - 1, sub_topic_num, sub_doc_len])

        n_d_topics_label[:, :, :] = n_d_topics[:].reshape(1, sub_topic_num, 1) * n_d_topics_label
        sub_eta = sub_eta_labels[:][:, :]
        hybrid_ws = (1 - sub_eta) * labeled_n_ws_topics[1:, :, :]
        hybrid_anchors = sub_eta * labeled_n_anchors_topics[1:, :].reshape(sub_label_num - 1, sub_topic_num, 1)
        hybrid_ws_topics[:, :, :] = hybrid_ws + hybrid_anchors
        hybrid_total_topics[:, :, :] = (1 - sub_eta) * labeled_total_n_topics[1:, :].reshape(sub_label_num - 1,
                                                                                             sub_topic_num, 1) \
                                       + sub_eta * labeled_n_anchors_totals_topics[1:, :].reshape(sub_label_num - 1,
                                                                                                  sub_topic_num, 1)

        prob_list = (hybrid_ws_topics + sub_beta) / (hybrid_total_topics + sub_words_num * sub_beta)
        topics_probs = self.p_get_topic_sel_probs(prob_list)
        topics_probs = (sub_label_topic_distributions.reshape(sub_label_num - 1, sub_topic_num, 1)
                        * sub_pred_label_probs).sum(0).reshape(1, sub_topic_num) * topics_probs
        p_d_w_topics_labels = (n_d_topics_label + sub_alpha) * (hybrid_ws_topics + sub_beta) / (
                hybrid_total_topics + sub_words_num * sub_beta)
        p_d_w_topics = p_d_w_topics_labels * sub_pred_label_probs
        new_topic_distributions = p_d_w_topics.sum(axis=0).T * topics_probs
        new_topic_distributions = new_topic_distributions / new_topic_distributions.sum(1).reshape(sub_doc_len, 1)
        for i in range(len(new_topic_distributions)):
            dist = new_topic_distributions[i]
            if len(dist[np.isnan(dist)]) > 0:
                print("error:", topics_probs.T[i], p_d_w_topics.sum(axis=0).T[i])
        return new_topic_distributions

    # gibbs_sampling iteration
    def p_gibbs_sampling(self,
                         sub_docs_list_blocks,
                         sub_labeled_topic_word,
                         sub_pre_label_list_blocks,
                         sub_pred_label_probs_blocks,
                         sub_label_num,
                         block_id,
                         sub_topic_num,
                         sub_words_num,
                         sub_doc_anchors_dict,
                         sub_alpha,
                         sub_beta,
                         q):
        """
            Function description: gibbs sampling main procedure.
            Parameters: (1) sub_docs_list_blocks: parallelized text data;
                        (2) sub_labeled_topic_word: class-specific topic-word frequencies;
                        (3) sub_pre_label_list_blocks: parallelized predicted labels for each word;
                        (4) sub_pred_label_probs_blocks: parallelized probabilities of labels for each word;
                        (5) sub_label_num: label numbers;
                        (6) block_id: id for paralleized data blocks;
                        (7) sub_topic_num: topic numbers;
                        (8) sub_words_num: numbers of words in total;
                        (12) sub_doc_anchors_dict: dictionary of anchor words for each document;
                        (13) sub_alpha: hyper-parameter of document-topic distribution;
                        (14) sub_beta: hyper-parameter of topic-word distribution;
                        (15) q: queue for parallelized algorithm.
            Return:
            Modification Record:
        """
        sub_docs_list = sub_docs_list_blocks[block_id]
        sub_pred_label_probs_list = sub_pred_label_probs_blocks[block_id]
        label_topic_distributions = sub_labeled_topic_word[1:].sum(axis=2) + sub_beta
        label_topic_distributions = label_topic_distributions / \
                                    label_topic_distributions.sum(axis=1).reshape(len(label_topic_distributions), 1)
        for d in range(0, len(sub_docs_list)):
            sub_pred_label_probs = sub_pred_label_probs_list[d]
            sub_pred_label = sub_pre_label_list_blocks[block_id][d]
            n_d_topics = self.p_get_n_d_topics(d, block_id, sub_docs_list_blocks, sub_topic_num)
            labeled_total_n_topics = self.p_get_labeled_total_n_topics(
                sub_labeled_topic_word.copy())  # (label_num, topic_num)

            doc_anchors_poses = sub_doc_anchors_dict[block_id * self.block_len + d]
            doc_anchors_idxs = sub_docs_list[d][doc_anchors_poses, 0]
            doc_anchors_idxs = [int(idx) for idx in doc_anchors_idxs]
            labeled_n_anchors_topics, labeled_n_anchors_totals_topics = self.p_get_labeled_anchors_k(doc_anchors_idxs,
                                                                                                     sub_labeled_topic_word.copy())

            labeled_n_ws_topics = self.p_get_labeled_n_ws_topics(d, block_id, sub_docs_list_blocks,
                                                                 sub_labeled_topic_word.copy())
            sub_eta = self.get_d_w_etas(labeled_n_ws_topics,
                                        labeled_total_n_topics,
                                        labeled_n_anchors_topics,
                                        labeled_n_anchors_totals_topics,
                                        sub_beta)
            sub_doc_len = len(sub_docs_list[d])
            new_pdfs = self.p_recompute_d_topic_distribution(label_topic_distributions,
                                                             sub_eta,
                                                             sub_pred_label_probs,
                                                             n_d_topics,
                                                             labeled_n_ws_topics,
                                                             labeled_total_n_topics,
                                                             labeled_n_anchors_topics,
                                                             labeled_n_anchors_totals_topics,
                                                             sub_words_num,
                                                             sub_topic_num,
                                                             sub_label_num,
                                                             sub_doc_len,
                                                             sub_alpha,
                                                             sub_beta)
            for w in range(0, len(sub_docs_list[d])):
                new_pdf = new_pdfs[w]
                new_topic = self.get_a_topic(new_pdf)
                sub_docs_list[d][w][1] = new_topic
            sub_docs_list[d][:, 3] = sub_pred_label * np.ones(sub_doc_len)
        q.put({block_id: sub_docs_list})

    def recompute_distributions(self):
        """
            Function description: recompute all distributions.
            Parameters:
            Return:
            Modification Record:
        """
        for d in range(0, len(self.doc_topic)):
            self.doc_topic_distributions[d] = (self.doc_topic[d] + self.alpha) / (
                    np.sum(self.doc_topic[d]) + len(self.doc_topic[d]) * self.alpha)
        for topic in range(0, len(self.topic_word)):
            self.topic_word_distribution[topic] = (self.topic_word[topic] + self.beta) / (
                    np.sum(self.topic_word[topic]) + len(self.topic_word[topic]) * self.beta)
        for lab in range(len(self.labeled_topic_word)):
            for topic in range(0, len(self.topic_word)):
                self.labeled_topic_word_distributions[lab][topic] \
                    = (self.labeled_topic_word[lab][topic] + self.beta) / (np.sum(self.labeled_topic_word[lab][topic])
                                                                           + len(
                            self.labeled_topic_word[lab][topic]) * self.beta)

    def compute_perplexities(self):
        """
            Function description: compute the perplexity of the training dataset.
            Parameters:
            Return:
            Modification Record:
        """
        self.doc_topic_distributions = np.array(self.doc_topic_distributions)
        self.labeled_topic_word_distributions = np.array(self.labeled_topic_word_distributions)
        self.topic_word_distribution = np.array(self.topic_word_distribution)
        total = 0
        total_num = 0
        acc = utils.compute_acc(self.pre_label_list[int(0.7 * len(self.docs_list)):],
                                self.labels_true[int(0.7 * len(self.docs_list)):])
        print("predicted accuracy of word labels:%.2f" % acc)
        for d in range(0, len(self.docs_list)):
            label = self.labels_true[d]
            for v in range(0, len(self.docs_list[d])):
                tw = self.labeled_topic_word_distributions[label]
                w = int(self.docs_list[d][v][0])
                p_d_w_k = tw[:, w]
                theta_d = self.doc_topic_distributions[d, :]
                total_t = (theta_d * p_d_w_k).sum(0)
                total += (-1) * math.log(total_t)
                total_num += 1.0
        return math.exp(total / total_num)

    @staticmethod
    def merge_dicts(dicts):
        new_dict = dict()
        for dic in dicts:
            new_dict.update(dic)
        return new_dict

    @staticmethod
    def dict_to_list(dic):
        new_list = []
        keys = sorted(list(dic.keys()))
        for k in keys:
            new_list += dic[k]
        return new_list

    def p_parameter_estimation(self):
        """
            Function description: parameter estimation main procedure.
            Parameters:
            Return:
            Modification Record:
        """
        self.per_list = list(self.per_list)
        self.per_list.clear()
        print(self.model_name)
        for i in range(0, self.round_num):
            st = time.time()
            q = multiprocessing.Queue()
            process_list = []
            for j in range(0, self.p_num):
                p = multiprocessing.Process(target=self.p_gibbs_sampling,
                                            args=(
                                                self.docs_list_blocks,
                                                self.labeled_topic_word,
                                                self.pre_label_list_blocks,
                                                self.pre_label_probs_list_blocks,
                                                self.label_num,
                                                j,
                                                self.topic_num,
                                                self.words_num,
                                                self.docs_anchor_words_dict,
                                                self.alpha,
                                                self.beta,
                                                q
                                            ))
                p.start()
                process_list.append(p)
            results = []
            for res in process_list:
                while res.is_alive():
                    while not q.empty():
                        results.append(q.get())
            for res in process_list:
                res.join()

            print("----TIME----:", time.time() - st)
            self.docs_list_blocks = self.merge_dicts([res for res in results])
            self.docs_list = self.dict_to_list(self.docs_list_blocks)
            self.compute_doc_topic()
            self.compute_topic_word()
            self.compute_labeled_topic_word()
            self.recompute_distributions()
            self.refresh_docs_labels()
            self.computer_words_anchor_scores()
            self.compute_docs_anchor_words_dict(self.tau)
            self.p_compute_pred_labels()
            self.per_list.append(self.compute_perplexities())
            print(i+1, "/", self.round_num, "perplexity: ", self.per_list[-1])
            self.temp_save(self.save_path)
        return

    def temp_save(self, path):
        """
            Function description: save the results of topic modeling for each sample iteration.
            Parameters:
            Return:
            Modification Record:
        """
        if not os.path.exists(path):
            os.makedirs(path)
        LDA_docs_list = np.array(self.docs_list)
        LDA_doc_topic_distributions = np.array(self.doc_topic_distributions)
        LDA_topic_word_distribution = np.array(self.topic_word_distribution)
        LDA_topic_word = np.array(self.topic_word)
        LDA_labeled_topic_word_distribution = np.array(self.labeled_topic_word_distributions)
        LDA_pre_label_probs_list = np.array(self.pre_label_probs_list)
        LDA_labeled_topic_word = np.array(self.labeled_topic_word)
        np.save(path + str(self.model_name) + "labeled_topic_word" + str(self.topic_num) + ".npy",
                LDA_labeled_topic_word)
        np.save(path + str(self.model_name) + "pre_label_probs_list" + str(self.topic_num) + ".npy",
                LDA_pre_label_probs_list)
        np.save(path + str(self.model_name) + "topic_word" + str(self.topic_num) + ".npy", LDA_topic_word)
        np.save(path + str(self.model_name) + "docs_list" + str(self.topic_num) + ".npy", LDA_docs_list)
        np.save(path + str(self.model_name) + "doc_topic_distributions_" + str(self.topic_num) + ".npy",
                LDA_doc_topic_distributions)
        np.save(path + str(self.model_name) + "topic_word_distribution_" + str(self.topic_num) + ".npy",
                LDA_topic_word_distribution)
        np.save(path + str(self.model_name) + "labeled_topic_word_distribution_" + str(self.topic_num) + ".npy",
                LDA_labeled_topic_word_distribution)
        LDA_per_list = np.array(self.per_list)
        np.save(path + str(self.model_name) + "per_list" + str(self.topic_num) + ".npy", LDA_per_list)
        f = open(path + str(self.model_name) + "docs_anchor_words_dict" + str(self.topic_num) + ".txt", 'w',
                 encoding=self.code_mode)
        f.write(str(self.docs_anchor_words_dict))
        f.close()
        return

    def save_result(self, path):
        """
            Function description: save the final results of topic modeling.
            Parameters:
            Return:
            Modification Record:
        """
        if not os.path.exists(path):
            os.makedirs(path)
        LDA_docs_list = np.array(self.docs_list)
        LDA_doc_topic_distributions = np.array(self.doc_topic_distributions)
        LDA_topic_word_distribution = np.array(self.topic_word_distribution)
        LDA_topic_word = np.array(self.topic_word)
        LDA_labeled_topic_word_distribution = np.array(self.labeled_topic_word_distributions)
        LDA_pre_label_probs_list = np.array(self.pre_label_probs_list)
        LDA_labeled_topic_word = np.array(self.labeled_topic_word)
        np.save(path + str(self.model_name) + "labeled_topic_word" + str(self.topic_num) + ".npy",
                LDA_labeled_topic_word)
        np.save(path + str(self.model_name) + "pre_label_probs_list" + str(self.topic_num) + ".npy",
                LDA_pre_label_probs_list)
        np.save(path + str(self.model_name) + "topic_word" + str(self.topic_num) + ".npy", LDA_topic_word)
        np.save(path + str(self.model_name) + "docs_list" + str(self.topic_num) + ".npy", LDA_docs_list)
        np.save(path + str(self.model_name) + "doc_topic_distributions_" + str(self.topic_num) + ".npy",
                LDA_doc_topic_distributions)
        np.save(path + str(self.model_name) + "topic_word_distribution_" + str(self.topic_num) + ".npy",
                LDA_topic_word_distribution)
        np.save(path + str(self.model_name) + "labeled_topic_word_distribution_" + str(self.topic_num) + ".npy",
                LDA_labeled_topic_word_distribution)
        LDA_per_list = np.array(self.per_list)
        np.save(path + str(self.model_name) + "per_list" + str(self.topic_num) + ".npy", LDA_per_list)
        f = open(path + str(self.model_name) + "word_scores_dict" + str(self.topic_num) + ".txt", 'w',
                 encoding=self.code_mode)
        f.write(str(self.word_scores_dict))
        f.close()
        f = open(path + str(self.model_name) + "docs_anchor_words_dict" + str(self.topic_num) + ".txt", 'w',
                 encoding=self.code_mode)
        f.write(str(self.docs_anchor_words_dict))
        f.close()
        f = open('word_index.txt', 'w', encoding=self.code_mode)
        f.write(str(self.word_index))
        f.close()
        f = open('index_word.txt', 'w', encoding=self.code_mode)
        f.write(str(self.index_word))
        f.close()
        return

    def train(self):
        """
            Function description: main procedure of the training for topic modeling.
            Parameters:
            Return:
            Modification Record:
        """
        self.p_parameter_estimation()
        self.save_result(self.save_path)
        return
