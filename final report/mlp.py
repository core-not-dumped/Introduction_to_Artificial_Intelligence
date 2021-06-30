# -*- coding: utf-8 -*-

from hw6_util import *

class Preprocessing(AI_util):
    def Calculate_Binary(self, data: List[Tuple[str, List[str], int]]) -> List[Tuple[str, List[float], int]]:

      binary_all = list()
      for (list_id,tokenized_text,cate) in data:
        binary_tmp = list(0 for i in range(len(self.word2idx)))
        for append_word in tokenized_text:
          if append_word in self.word2idx.keys():
            binary_tmp[self.word2idx[append_word]] = 1
        binary = (list_id, binary_tmp, cate)
        binary_all.append(binary)

      return binary_all

    def Calculate_TF(self, data: List[Tuple[str, List[str], int]]) -> List[Tuple[str, List[float], int]]:

      tf_all = list()
      for (list_id,tokenized_text,cate) in data:
        tf_tmp = list(0 for i in range(len(self.word2idx)))
        for append_word in tokenized_text:
          if append_word in self.word2idx.keys():
            tf_tmp[self.word2idx[append_word]] += 1
        tf = (list_id,tf_tmp,cate)
        tf_all.append(tf)

      return tf_all

    def Calculate_TF_IDF_Normalization(self, data: List[Tuple[str, List[str], int]], data_type: str) -> List[Tuple[str, List[float], int]]:

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

class MLP:
    def __init__(self, input_size: int, hidden_size: int, output_size: int, learning_rate: float):
      ### EDIT HERE ###
      self.input_size = input_size
      self.hidden_size = hidden_size
      self.output_size = output_size

      self.input_layer = np.zeros(input_size)
      self.hidden_layer = np.zeros(hidden_size)
      self.output_layer = np.zeros(output_size)

      self.learning_rate = learning_rate
      
      self.w1 = (np.random.rand(input_size, hidden_size)-0.5) * 2 * math.sqrt(6/(input_size + hidden_size))
      self.w2 = (np.random.rand(hidden_size, output_size)-0.5) *  2 * math.sqrt(6/(output_size + hidden_size))

      self.del_w1 = np.zeros((input_size, hidden_size))      
      self.del_w2 = np.zeros((hidden_size, output_size))  

      self.w1_b = np.zeros(hidden_size)
      self.w2_b = np.zeros(output_size)

      self.del_w1_b = np.zeros(hidden_size)
      self.del_w2_b = np.zeros(output_size)

      self.feature_scale = 1  

      self.preds = np.zeros(output_size)
      self.real = np.zeros(output_size)

      self.z_sum = np.zeros(hidden_size)
      self.o_sum = np.zeros(output_size)
      ### END ###

    def forward(self, input):
      # This function is for forwarding.
      ### EDIT HERE ###
      self.preds = np.zeros(self.output_size)
      self.real = np.zeros(self.output_size)

      self.input_layer = np.zeros(self.input_size)
      self.hidden_layer = np.zeros(self.hidden_size)
      self.output_layer = np.zeros(self.output_size)

      self.z_sum = np.zeros(self.hidden_size)
      self.o_sum = np.zeros(self.output_size)

      # input_layer
      for i in range(self.input_size):
        self.input_layer[i] = input[1][i] * self.feature_scale

      # hidden_layer
      for i in range(self.hidden_size):
        for j in range(self.input_size):
          self.z_sum[i] += (self.input_layer[j] * self.w1[j][i])
        self.z_sum[i] += self.w1_b[i]
      self.hidden_layer = np.tanh(self.z_sum)

      # output_layer
      for i in range(self.output_size):
        for j in range(self.hidden_size):
          self.o_sum[i] += (self.hidden_layer[j] * self.w2[j][i])
        self.o_sum[i] += self.w2_b[i]
      self.output_layer = np.exp(self.o_sum)/np.sum(np.exp(self.o_sum))

      self.real[int(input[2])] = 1
      self.preds = self.output_layer
      print(np.round(self.real,2))
      print(np.round(self.preds,2))

      return np.argmax(self.preds)
      ### END ###

    def backward(self):
      # This function is for back propagation.
      ### EDIT HERE ###

      # (t-o)tau'(o_sum)
      delta = self.o_sum
      softmax = np.exp(delta)/np.sum(np.exp(delta))
      delta = softmax * (1-softmax)
      delta = (self.real - self.preds) * delta

      # alpha * delta * hidden
      del_tmp2 = self.learning_rate * delta
      for o in range(self.output_size):
        for z in range(self.hidden_size):
          self.del_w2[z][o] += (del_tmp2[o] * self.hidden_layer[z])

      self.del_w2_b += del_tmp2

      # tau'(z_sum)
      del_tmp1 = self.z_sum
      del_tmp1 = (1 + np.tanh(del_tmp1)) * (1 - np.tanh(del_tmp1))

      # tau'(z_sum) * sigma(delta * v)
      for z in range(self.hidden_size):
        sum_for_del_tmp1 = 0
        for o in range(self.output_size):
          sum_for_del_tmp1 += delta[o] * self.w2[z][o]
        del_tmp1[z] *= sum_for_del_tmp1

      # alpha * etha * input
      del_tmp1[z] *= self.learning_rate
      for z in range(self.hidden_size):
        for i in range(self.input_size):
          self.del_w1[i][z] += del_tmp1[z] * self.input_layer[i]

      self.del_w1_b += del_tmp1
      
      ### END ###

    def step(self):
      # This function is for weight updating.
      ### EDIT HERE ###
      self.w1 += self.del_w1
      self.w2 += self.del_w2
      self.w1_b += self.del_w1_b
      self.w2_b += self.del_w2_b
      ### END ###

    def loss(self):
      # This function is for calculating loss between logits and labels.
      ### EDIT HERE ###
      print('loss')
      ### END ###


def main(data, label2idx):
    ### EDIT HERE ###
    std_name = "이승태"
    std_id = "2017313107"    
    ### END ###
    result = dict()
    for inp_type, tr, te in tqdm(data, desc='training & evaluating...'):
      """
          This function is for training and evaluating (testing) MLP Model.
      """

      ### EDIT HERE ###
      epoch = 2
      mlp = MLP(len(tr[0][1]), 30, 5, 0.01)
      if inp_type == 'TF':
        mlp = MLP(len(tr[0][1]), 40, 5, 0.01)
      if inp_type == 'TF-IDF':
        mlp = MLP(len(tr[0][1]), 35, 5, 0.01)
        mlp.feature_scale = 100

      for i in range(epoch):
        accuracy = 0.0
        for tr_idx in range(len(tr)):
          pred = mlp.forward(tr[tr_idx])
          print(tr_idx, pred, tr[tr_idx][2])
          if pred == tr[tr_idx][2]:
            accuracy += 1
          mlp.backward()
          if tr_idx%10 == 9:
            mlp.step()
            mlp.del_w1 = np.zeros((mlp.input_size, mlp.hidden_size))      
            mlp.del_w2 = np.zeros((mlp.hidden_size, mlp.output_size))  
            mlp.del_w1_b = np.zeros(mlp.hidden_size)
            mlp.del_w2_b = np.zeros(mlp.output_size) 
        accuracy = accuracy / len(tr)
        print('accuracy:')
        print(accuracy)

      result[inp_type] = dict()
      accur_map = np.zeros((5,5))
      accuracy = 0.0
      number_of_docs = np.zeros(5)
      for te_idx in range(len(te)):

        number_of_docs[te[te_idx][2]] += 1

        pred = mlp.forward(te[te_idx])
        print(te_idx, pred, te[te_idx][2])
        accur_map[pred][te[te_idx][2]] += 1
        if pred == te[te_idx][2]:
          accuracy += 1
      accuracy = accuracy / len(te)
      print('accuracy:')
      print(accuracy)

      micro_map = np.zeros((2,2))
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
        pre = TP / (FP + TP)
        re = TP / (FN + TP)
        f1 = 2 * pre * re / (pre + re)
        
        result[inp_type][label[i]] = (pre * 100, re * 100, f1 * 100, number_of_docs[i])

      result[inp_type]['accuracy'] = (accuracy * 100, np.sum(number_of_docs))

      print(micro_map)
      pre = micro_map[0][0] / (micro_map[0][0] + micro_map[1][0])
      re = micro_map[0][0] / (micro_map[0][0] + micro_map[0][1])
      f1 = 2 * pre * re / (pre + re)
      result[inp_type]['micro avg'] = (pre * 100, re * 100, f1 * 100, np.sum(number_of_docs))

      print(result)

      ### END ###

    """
      result(input variable for "save_result" function) contains 
            1. Performance for each labels (precision, recall, f1-score per label)
            2. Overall micro average and accuracy for the entire test dataset
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
                "micro avg": (precision, recall, f1-score, total docs)
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
    train_data = Preprocessing.load_data(data_path='./train.json', data_type='train')
    Preprocessing.tr_binary = Preprocessing.Calculate_Binary(data=train_data)
    Preprocessing.tr_tf = Preprocessing.Calculate_TF(data=train_data)
    Preprocessing.tr_tfidf = Preprocessing.Calculate_TF_IDF_Normalization(data=train_data, data_type='train')
    test_data = Preprocessing.load_data(data_path='./test.json', data_type='test')
    Preprocessing.te_binary = Preprocessing.Calculate_Binary(data=test_data)
    Preprocessing.te_tf = Preprocessing.Calculate_TF(data=test_data)
    Preprocessing.te_tfidf = Preprocessing.Calculate_TF_IDF_Normalization(data=test_data, data_type='test')

    data = [
        ('Binary', Preprocessing.tr_binary, Preprocessing.te_binary), 
        ('TF', Preprocessing.tr_tf, Preprocessing.te_tf), 
        ('TF-IDF', Preprocessing.tr_tfidf, Preprocessing.te_tfidf)
        ]

    main(data, Preprocessing.label2idx)
    #   *** Do not modify the code above ***
