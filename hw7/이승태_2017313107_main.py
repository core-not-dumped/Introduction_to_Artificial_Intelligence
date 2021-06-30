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
      binary_all = list()
      for (list_id,tokenized_text,cate) in data:
        binary_tmp = list(0 for i in range(len(self.word2idx)))
        for append_word in tokenized_text:
          if append_word in self.word2idx.keys():
            binary_tmp[self.word2idx[append_word]] = 1
        binary = (list_id, binary_tmp, cate)
        binary_all.append(binary)

      return binary_all

    def Calculate_TF(self, data: List[Tuple[str, List[str], int]])  -> List[Tuple[str, List[float], int]]:
      """
        *** You should implement this function with raw code ***
        *** When you code, you have to erase this comment ***
        (input) 'data' type : List[Tuple[str, List[str], int]]
        (input) 'data' format :   [(document id, tokenized text, category index)]

        (output) return type : List[Tuple[str, List[float], int]]
        (output) return format : [(document id, TF, category index)]           
      """
      tf_all = list()
      for (list_id,tokenized_text,cate) in data:
        tf_tmp = list(0 for i in range(len(self.word2idx)))
        for append_word in tokenized_text:
          if append_word in self.word2idx.keys():
            tf_tmp[self.word2idx[append_word]] += 1
        tf = (list_id,tf_tmp,cate)
        tf_all.append(tf)

      return tf_all

    def Calculate_TF_IDF_Normalization(self, data: List[Tuple[str, List[str], int]], data_type: str)  -> List[Tuple[str, List[float], int]]:
      """
        *** You should implement this function with raw code ***
        *** When you code, you have to erase this comment ***
        (input) 'data' type : List[Tuple[str, List[str], int]]
        (input) 'data' format :   [(document id, tokenized text, category index)]

        (output) return type : List[Tuple[str, List[float], int]]
        (output) return format : [(document id, normalized tf-idf, category index)]           
      """
      tf = list()
      for (list_id,tokenized_text,cate) in data:
        tf_tmp = list(0 for i in range(len(self.word2idx)))
        for append_word in tokenized_text:
          if append_word in self.word2idx.keys():
            tf_tmp[self.word2idx[append_word]] += 1
        tf.append(tf_tmp)

      if len(data)>200:
        self.idf = list(0 for i in range(len(self.word2idx)))
        for (list_id,tokenized_text,cate) in data:
          idf_tmp = list(0 for i in range(len(self.word2idx)))
          for append_word in tokenized_text:
            if append_word in self.word2idx.keys():
              idf_tmp[self.word2idx[append_word]] = 1
          for i in range(len(self.word2idx)):
            self.idf[i] += idf_tmp[i]

        for i in range(len(self.word2idx)):
          self.idf[i] = math.log(float(len(data))/float(self.idf[i]),2)

      tfidf_all = list()
      for i in range(len(tf)):
        tfidf_tmp = list(0 for j in range(len(self.word2idx)))
        tfidf_sqare_sum_tmp = 0
        for j in range(len(self.word2idx)):
          tfidf_tmp[j] = tf[i][j] * self.idf[j]
          tfidf_sqare_sum_tmp += (tfidf_tmp[j] * tfidf_tmp[j])
        for j in range(len(self.word2idx)):
          tfidf_tmp[j] = tfidf_tmp[j] / math.sqrt(tfidf_sqare_sum_tmp)
        tfidf = (data[i][0], tfidf_tmp, data[i][2])
        if i == 0:
          print(tfidf)
        tfidf_all.append(tfidf)

      return tfidf_all

def main(data, label2idx):

    std_name = "이승태"
    std_id = "2017313107"
    result = dict()
    for inp_type, tr, te in tqdm(data, desc='training & evaluating...'):
      """
        This function is for training and evaluating (testing) SVM Model.
      """
      ### EDIT HERE ###
      train_inputs = list()
      train_label = list()
      test_inputs = list()
      test_label = list()

      for i in range(len(tr)):
        train_inputs.append(tr[i][1])
        train_label.append(tr[i][2])

      for i in range(len(te)):
        test_inputs.append(te[i][1])
        test_label.append(te[i][2])
      
      classifier = LinearSVC(C=1.0,max_iter=1000)
      classifier.fit(train_inputs, train_label)

      prediction = classifier.predict(test_inputs)

      result[inp_type] = dict()
      accur_map = np.zeros((5,5))
      accuracy = 0.0
      number_of_docs = np.zeros(5)
      for i in range(len(te)):
        number_of_docs[test_label[i]] += 1
        accur_map[prediction[i]][test_label[i]] += 1
        if prediction[i] == test_label[i]:
          accuracy += 1
      accuracy = accuracy / len(te)
      print('accuracy:')
      print(accuracy)

      micro_map = np.zeros((2,2))
      precision = np.zeros(5)
      recall = np.zeros(5)
      label = {0:'entertainment',1:'finance',2:'lifestyle',3:'sports',4:'tv'}
      for i in range(5):
        TP = accur_map[i][i]
        FN = 0.0
        FP = 0.0
        for j in range(5):
          if i==j:
            continue
          FP += accur_map[i][j]
        for j in range(5):
          if i==j:
            continue
          FN += accur_map[j][i]

        micro_map[0][0] += TP
        micro_map[0][1] += FN
        micro_map[1][0] += FP
        precision[i] = TP / (FP + TP)
        recall[i] = TP / (FN + TP)
        f1 = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
        
        result[inp_type][label[i]] = (precision[i] * 100, recall[i] * 100, f1 * 100, number_of_docs[i])

      result[inp_type]['accuracy'] = (accuracy * 100, np.sum(number_of_docs))

      pre = micro_map[0][0] / (micro_map[0][0] + micro_map[1][0])
      re = micro_map[0][0] / (micro_map[0][0] + micro_map[0][1])
      f1 = 2 * pre * re / (pre + re)
      result[inp_type]['micro avg'] = (pre * 100, re * 100, f1 * 100, np.sum(number_of_docs))

      pre = np.sum(precision)/5
      re = np.sum(recall)/5
      result[inp_type]['macro avg'] = (pre * 100, re * 100, f1 * 100, np.sum(number_of_docs))

      print(result)

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