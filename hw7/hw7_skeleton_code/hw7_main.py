# -*- coding: utf-8 -*-

from hw7_util import *

class Preprocessing(AI_util):
    def Calculate_Binary(self, data: List[Tuple[str, List[str], int]])  -> List[Tuple[str, List[float], int]]:
        """
            *** You should implement this function with raw code ***
            *** When you code, you have to erase this comment ***
            (input) 'data' type : List[Tuple[str, List[str], int]]
            (input) 'data' format :   [(document id, tokenized text, category index)]

            (output) return type : List[Tuple[str, List[float], int]]
            (output) return format : [(document id, Binary vector, category index)]           
        """  
        return

    def Calculate_TF(self, data: List[Tuple[str, List[str], int]])  -> List[Tuple[str, List[float], int]]:
        """
            *** You should implement this function with raw code ***
            *** When you code, you have to erase this comment ***
            (input) 'data' type : List[Tuple[str, List[str], int]]
            (input) 'data' format :   [(document id, tokenized text, category index)]

            (output) return type : List[Tuple[str, List[float], int]]
            (output) return format : [(document id, TF, category index)]           
        """  
        return

    def Calculate_TF_IDF_Normalization(self, data: List[Tuple[str, List[str], int]], data_type: str)  -> List[Tuple[str, List[float], int]]:
        """
            *** You should implement this function with raw code ***
            *** When you code, you have to erase this comment ***
            (input) 'data' type : List[Tuple[str, List[str], int]]
            (input) 'data' format :   [(document id, tokenized text, category index)]

            (output) return type : List[Tuple[str, List[float], int]]
            (output) return format : [(document id, normalized tf-idf, category index)]           
        """  
        return

def main(data, label2idx):

    std_name = "GildongHong"
    std_id = "2021123456"
    result = dict()
    for inp_type, tr, te in tqdm(data, desc='training & evaluating...'):
        """
            This function is for training and evaluating (testing) SVM Model.
        """
        ### EDIT HERE ###

        ### END ###
    """
        result(input variable for "save_result" function) contains 
            1. Performance for each labels (precision, recall, f1-score per label)
            2. Overall micro/macro average and accuracy for the entire test dataset
            3. Convert the result 1 and 2 into percentages by multiplying 100

        result type : Dict[str, Dict[str, Union[Tuple[float, float, float, int], Tuple[float, int]]]]
        result input format for "save_result" function: 
        {
            'Binary': 
            {
                "entertainment": (precision, recall, f1-score, # of docs),
                "finance": (precision, recall, f1-score, # of docs),
                "lifestyle": (precision, recall, f1-score, # of docs),
                "sports": (precision, recall, f1-score, # of docs),
                "tv": (precision, recall, f1-score, # of docs),
                "accuracy": (accuracy, total docs),
                "micro avg": (precision, recall, f1-score, total docs),
                "macro avg": (precision, recall, f1-score, total docs)
            },
            "TF": ...,
            "TF-IDF": ...,
        }
    """

    save_result(result, std_name=std_name, std_id=std_id)

if __name__ == "__main__":
    #   *** Do not modify the code below ***
    random.seed(42)
    np.random.seed(42)

    Preprocessing = Preprocessing()
    tr_data = Preprocessing.load_data(data_path='./train.json', data_type='train')
    Preprocessing.tr_binary = Preprocessing.Calculate_Binary(data=tr_data)
    Preprocessing.tr_tf = Preprocessing.Calculate_TF(data=tr_data)
    Preprocessing.tr_tfidf = Preprocessing.Calculate_TF_IDF_Normalization(data=tr_data, data_type='train')
    te_data = Preprocessing.load_data(data_path='./test.json', data_type='test')
    Preprocessing.te_binary = Preprocessing.Calculate_Binary(data=te_data)
    Preprocessing.te_tf = Preprocessing.Calculate_TF(data=te_data)
    Preprocessing.te_tfidf = Preprocessing.Calculate_TF_IDF_Normalization(data=te_data, data_type='test')

    data = [
        ('Binary', Preprocessing.tr_binary, Preprocessing.te_binary), 
        ('TF', Preprocessing.tr_tf, Preprocessing.te_tf), 
        ('TF-IDF', Preprocessing.tr_tfidf, Preprocessing.te_tfidf)
        ]

    main(data, Preprocessing.label2idx)
    #   *** Do not modify the code above ***