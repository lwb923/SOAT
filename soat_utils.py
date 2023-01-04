from scipy.stats import entropy
import heapq
from collections import Counter


# conpute the accuracy for predicted labels for all words in documents
def compute_acc(list1, list2):
    if len(list1) != len(list2):
        return -1
    score = 0
    for i in range(len(list1)):
        if list1[i] == list2[i]:
            score += 1
    return score / len(list1)


# split the original data into labeled data and unlabeled data according to the given percentage
def filter_labels(labels, percentage):
    labels_len = int(percentage * len(labels))
    labels[labels_len:] = [0 for item in labels[labels_len:]]

    return labels


# compute the class-specific anchor words for a given word
def get_word_labeled_anchors(word,
                             docs_list,
                             docs_anchor_words_dict,
                             true_label_list,
                             SOAT_word2idx,
                             SOAT_idx2word,
                             label_num,
                             topk=20):
    w_id = SOAT_word2idx[word]
    word_anchor_dict = dict()
    for i in range(1, label_num + 1):
        word_anchor_dict[i] = []
    for d in range(len(docs_list)):
        words = docs_list[d][:, 0]
        words = [int(w) for w in words]
        if w_id not in words:
            continue
        anchors = list(docs_anchor_words_dict[d])
        word_anchor_dict[true_label_list[d]] += [SOAT_idx2word[words[w]] for w in anchors if w is not Ellipsis]
    for label in range(1, label_num + 1):
        word_anchor_dict[label] = Counter(word_anchor_dict[label]).most_common(topk)
    return word_anchor_dict


# compute the class-specific topical words
def get_label_words(l_topic_word, l_topic_word_dist, idx2word, topk, beta=0.05):
    label_topic_dist = l_topic_word[:].sum(axis=2) + beta
    label_topic_dist = label_topic_dist / \
                       label_topic_dist.sum(axis=1).reshape(len(label_topic_dist), 1)
    label_words_list = []
    for i in range(len(label_topic_dist)):
        label_word_prob = label_topic_dist[i].dot(l_topic_word_dist[i])
        candidates_idxs = heapq.nlargest(topk, range(len(label_word_prob)), label_word_prob.take)
        label_words = [idx2word[idx] for idx in candidates_idxs]
        label_words_list.append(label_words)
    return label_words_list
