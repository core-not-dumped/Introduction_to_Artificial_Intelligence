# -*- coding: utf-8 -*-

from hw3_util import *

class Preprocessing(AI_util):
    def Calculate_TF_IDF_Normalization(self, data: List[Tuple[str, List[str], str]])  -> List[Tuple[str, List[str], str]]:
        # train test data 
        if int(data[0][0]) < 800:
          train = 1
        else:
          train = 0

        # make sorted_word_list in train data
        if train == 1:
          word_list = set()
          for (list_id,tokenized_text,cate) in data:
            for append_word in tokenized_text:
              word_list.add(append_word)
          self.sorted_word_list = sorted(word_list)
          # for test
          # print(self.sorted_word_list)
        
        # make tf, idf list
        tf = list()
        if train == 1:
          self.idf = list(0 for i in range(len(self.sorted_word_list)))
        for (list_id,tokenized_text,cate) in data:
          tf_for_one = list(0 for i in range(len(self.sorted_word_list)))
          for article_in_word in tokenized_text:
            # find word in sroted_word_list
            if article_in_word in self.sorted_word_list:
              find_index = self.sorted_word_list.index(article_in_word)
            else:
              continue;
            
            # make tf_for_one list
            if train == 1:
              if tf_for_one[find_index] == 0:
                self.idf[find_index] += 1
            tf_for_one[find_index] += 1

          # append to tf
          tf.append(tf_for_one)
        # for test
        # print(self.idf);

        # make idf list
        if train == 1:
          data_number = len(data)
          for i in range(len(self.idf)):
            self.idf[i] = round(math.log(data_number / self.idf[i], 2),2)
        # for test
        # print(self.idf);

        # make tf_idf
        res = list()
        for i in range(len(tf)):
          tf_idf = list(0 for j in range(len(self.sorted_word_list)))
          for j in range(len(self.sorted_word_list)):
            tf_idf[j] = tf[i][j] * self.idf[j]
          sqare_sum = 0
          for num in tf_idf:
            sqare_sum += (num * num)
            root_sqare_sum = math.sqrt(sqare_sum)
          for j in range(len(tf_idf)):
            tf_idf[j] = round(tf_idf[j] / root_sqare_sum , 2)
          res.append([data[i][0],tf_idf,data[i][2]])

        #"""
        #    *** You should implement this function with raw code ***
        #    *** When you code, you have to erase this comment ***
        #    (input) 'data' type : ('list')
        #    (input) 'data' format :   [(id, tokenized text, category)]

        #    (output) return type : ('list')
        #    (output) return format : [(article id, normalized tf-idf, category)]           
        #"""
        
        return res

if __name__ == "__main__":
    #   *** Do not modify the code below ***
    parser = argparse.ArgumentParser()

    parser.add_argument("--document_id",
                        default=815,
                        type=int)
    args = parser.parse_args()

    Preprocessing = Preprocessing()
    train_data = Preprocessing.load_data(data_path='./train.json', data_type='train')
    Preprocessing.train_tfidf = Preprocessing.Calculate_TF_IDF_Normalization(data=train_data)
    test_data = Preprocessing.load_data(data_path='./test.json', data_type='test')
    Preprocessing.test_tfidf = Preprocessing.Calculate_TF_IDF_Normalization(data=test_data)

    Preprocessing.save_reulst(std_name="이승태", std_id ="2017313107", document_id=args.document_id)
    #   *** Do not modify the code above ***
