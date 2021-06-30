#   *** Do not import any library except already imported libraries ***

import argparse
import logging
import os
from pathlib import Path
import json

import math
import random
import numpy as np
from collections import Counter
from tqdm import tqdm, trange

from hw5_utils import AI_util
#   *** Do not import any library except already imported libraries ***

class KNN(AI_util):
    def __init__(self, k):
        super(KNN, self).__init__()
        self.k = k
    
    def predict(self, train_data, test_data):
        preds = None

        ### EDIT HERE ###
        if self.cnt >= 0:
          self.cnt += 1
        else:
          self.cnt = 0
        
        preds = list(0 for i in range(len(test_data)))

        if self.cnt <= 1 and self.cnt == 4:
          train_bow = self.get_bow(train_data)
          test_bow = self.get_bow(test_data)

          if self.cnt <= 1:
            for i in range(len(test_data)):
              ch_list = list(0 for i in range(self.n_labels))
              dis = list(0 for i in range(len(train_data)))
              for j in range(len(train_data)):
                dis[j] = get_cosine_sim(test_bow[i],train_bow[j])
              sorted(range(len(dis)), key=lambda k: dis[k])
              for j in range(self.k):
                label = train_data[dis[len(train_data) - j - 1][2]]
                ch_list[self.label2idx(label)] += 1
              preds[i] = ch_list.index(max(ch_list))

          
        else :
          train_tf = self.get_tf(train_data)
          test_tf = self.get_tf(test_data)

          train_idf = self.get_idf(train_data)

          train_tfidf = self.get_normalized_tfidf(train_tf,train_idf)
          test_tfidf = self.get_normalized_tfidf(test_tf,train_idf)

        ### END ###

        return preds
    
    def get_bow(self, documents):
        bow = None
        ### EDIT HERE ###

        bow = list()
        for (list_id,tokenized_text,cate) in documents:
          bow_tmp = list(0 for i in range(len(self.word2idx)))
          for append_word in tokenized_text:
            if append_word in self.word2idx.keys():
              bow_tmp[self.word2idx[append_word]] = 1
          bow.append(bow_tmp)

        ### END ###

        return bow

    def get_tf(self, documents):
        tf = None

        ### EDIT HERE ###
        
        tf = list()
        for (list_id,tokenized_text,cate) in documents:
          tf_tmp = list(0 for i in range(len(self.word2idx)))
          for append_word in tokenized_text:
            if append_word in self.word2idx.keys():
              tf_tmp[self.word2idx[append_word]] += 1
          tf.append(tf_tmp)

        ### END ###

        return tf

    def get_idf(self, documents):
        idf = None

        ### EDIT HERE ###
        idf = list(0 for i in range(len(self.word2idx)))
        for (list_id,tokenized_text,cate) in documents:
          idf_tmp = list(0 for i in range(len(self.word2idx)))
          for append_word in tokenized_text:
            if append_word in self.word2idx.keys():
              idf_tmp[self.word2idx[append_word]] = 1
          for i in range(len(self.word2idx)):
            idf[i] += idf_tmp[i]
        print(idf)

        for i in range(len(self.word2idx)):
          idf[i] = round(math.log(float(len(self.word2idx))/float(idf[i]),2),2)
        print(idf)

        ### END ###

        return idf

    def get_normalized_tfidf(self, tf, idf):
        tfidf = None

        ### EDIT HERE ###
        tfidf = list()
        for i in range(len(tf)):
          tfidf_tmp = list(0 for j in range(len(self.word2idx)))
          tfidf_sqare_sum_tmp = 0
          for j in range(len(self.word2idx)):
            tfidf_tmp[j] = tf[i][j] * idf[j]
            tfidf_sqare_sum_tmp += (tfidf_tmp[j] * tfidf_tmp[j])
          for j in range(len(self.word2idx)):
            tfidf_tmp[j] = tfidf_tmp[j] / math.sqrt(tfidf_sqare_sum_tmp)
          tfidf.append(tfidf_tmp)

        ### END ###

        return tfidf

    def get_euclidean_dist(self, vec_a, vec_b):
        dist = None

        ### EDIT HERE ###

        dist = 0
        for i in range(len(self.word2idx)):
          dist += (vec_a[i] - vec_b[i]) * (vec_a[i] - vec_b[i])
        
        ### END ###

        return dist
    
    def get_cosine_sim(self, vec_a, vec_b):
        sim = None

        ### EDIT HERE ###

        dist = 0
        for i in range(len(self.word2idx)):
          dist += (vec_a[i] * vec_b[i])

        ### END ###

        return sim


def main(args, logger):

    k = args.k
    knn = KNN(k)
    train_data = knn.load_data(args.data_dir, 'train')
    test_data = knn.load_data(args.data_dir, 'test')

    labels = [knn.label2idx[l] for _, _, l in test_data]
    preds = knn.predict(train_data, test_data)
    acc = knn.calc_accuracy(labels, preds)
    
    logger.info("Accuracy: {:.2f}%".format(acc * 100))

    ### EDIT ###
    """ Implement codes writing output file """

    std_name = '이승태'
    std_id = '2017313107'

    f = open("./{}_{}_hw5.txt".format(std_name,std_id), 'w')
    
    f.write("Metric: \n")
    f.write("Input: \n")
    f.write("Accuracy: {}%\n".format(str(acc * 100)))

    ### END ###
    if std_name == None: raise ValueError
    if std_id == None: raise ValueError

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',
                        type=str,
                        default='./',
                        help="Path where dataset exist.")
    parser.add_argument('--output_dir',
                        type=str,
                        default='./',
                        help="Path where output will be saved.")

    parser.add_argument('--k',
                        type=int,
                        default=11)
    args = parser.parse_args()

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
    logger = logging.getLogger(__name__)

    main(args, logger)
