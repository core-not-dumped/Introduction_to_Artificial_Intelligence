#   *** Do not import any library except already imported libraries ***

import argparse
import logging
import os
from pathlib import Path
import json

import math
import numpy as np
from tqdm import tqdm, trange
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from hw4_util import AI_util
#   *** Do not import any library except already imported libraries ***

class Naive_Bayes(AI_util):
    def __init__(self, data_path):
        super(Naive_Bayes, self).__init__()

    def predict(self, test_data: list):
        """
        In this function, you have to predict the labels of test dataset.
        Also, save article ids and values of predicted labels.
        Do not use labels in test_data. It's not for the predict function.
        """
        ids = list()        # article ids
        preds = list()      # predicted labels
        values = list()     # log probabilities of predicted labels

        ### EDIT FROM HERE ###
        
        ### EDIT UNTIL HERE ###

        return ids, values, preds
    
    def calc_conditional_prob(self, train_data: list):
        """
        In this function, you have to calculate conditional probabilities with training dataset.
        You can choose data structure of self.cond_prob as whatever you want to use.
        Then, you can use it in predict function.
        """
        self.cond_prob = None

        ### EDIT FROM HERE ###
        
        ### EDIT UNTIL HERE ###

    def get_tf_vector(self, document):
        """
        This function returns fixed size of TF vector for a document.
        You can use this function if you want to use.
        """
        tf_vec = np.zeros(len(self.word2idx), dtype=np.int32)
        
        for token in document:
            if token in self.word2idx.keys():
                tf_vec[self.word2idx[token]] += 1

        return tf_vec


def main(args, logger):
    data_path = os.path.join('./')

    nb_classifier = Naive_Bayes(data_path)
    logger.info("Classifier is initialized!")

    train_data = nb_classifier.load_data(data_path, 'train')
    logger.info("# of train data: {}".format(len(train_data)))
    test_data = nb_classifier.load_data(data_path, 'test')
    logger.info("# of test data: {}".format(len(test_data)))

    nb_classifier.calc_conditional_prob(train_data)
    ids, values, preds = nb_classifier.predict(test_data)
    labels = [nb_classifier.label2idx[y] for _, _, y in test_data]

    accuracy = accuracy_score(labels, preds)
    precsion = precision_score(labels, preds, average='micro')
    recall = recall_score(labels, preds, average='micro')
    f1 = f1_score(labels, preds, average='micro')

    logger.info("Accuracy: {}%".format(accuracy * 100))
    logger.info("Precision: {}%".format(precsion * 100))
    logger.info("Recall: {}%".format(recall * 100))
    logger.info("F1: {}%".format(f1 * 100))

    with open(args.output_dir/'{}_{}.txt'.format(args.std_name, args.std_id), 'w', encoding='utf-8') as f:
        f.write("Accuracy: {}%\n".format(str(accuracy * 100)))
        f.write("Precision: {}%\n".format(str(precsion * 100)))
        f.write("Recall: {}%\n".format(str(recall * 100)))
        f.write("F1: {}%\n\n".format(str(f1 * 100)))
        
        ### EDIT FROM HERE ###
        """ Implement codes for the output text file. """
        
        ### EDIT UNTIL HERE ###

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--std_name',
                        type=str,
                        default='홍길동',
                        help='Student name.')
    parser.add_argument('--std_id',
                        type=str,
                        default="2021321321",
                        help='Student ID')

    parser.add_argument('--selected_label',
                        type=str,
                        default='tv',
                        help="Label to write its scores down to output text file.")
    parser.add_argument('--output_dir',
                        type=Path,
                        default=Path('./'),
                        help="Path where output will be saved.")
    args = parser.parse_args()

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
    logger = logging.getLogger(__name__)

    main(args, logger)
