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

        ### END ###

        return preds
    
    def get_bow(self, documents):
        bow = None

        ### EDIT HERE ###

        ### END ###

        return bow

    def get_tf(self, documents):
        tf = None

        ### EDIT HERE ###
        
        ### END ###

        return tf

    def get_idf(self, documents):
        idf = None

        ### EDIT HERE ###

        ### END ###

        return idf

    def get_normalized_tfidf(self, tf, idf):
        tfidf = None

        ### EDIT HERE ###

        ### END ###

        return tfidf

    def get_euclidean_dist(self, vec_a, vec_b):
        dist = None

        ### EDIT HERE ###
        
        ### END ###

        return dist
    
    def get_cosine_sim(self, vec_a, vec_b):
        sim = None

        ### EDIT HERE ###

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
    std_name = None
    std_id = None

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
