import numpy as np
import pandas as pd
from collections import Counter
import re
import os
import sys

class data_loader:
    '''
    This class is used to load traing data from files

    There are three types of files acceepted: 

    1.  Ontonotes 5.0 data with Conll 2012 manually annotated format(The file's name ends with "gold_conll"), 
        The files should in the "\data\conll-2012\" folder with the original structure

    2.  The CSV file generated from the data mentioned above by this program
        The files should in the "\data\csv\" folder named with "train.csv", "test.csv" and "development.csv"

    3.  txt files generatated from the csv file mentioned above
        The files should in the "\data\text\" folder named with "train_word.txt", "train_tag.txt", "test_word.txt", "test_tag.txt", "development_word.txt", "development_tag.txt"
   
    You can also provide your own training data to train your own model with this program.
    By using your own data, you should put your .txt format data into the right folder with the right name,
    
    The format descipts below:

    In each "_word.txt" file contains one sentence per line, words(include punctuations) are seperated by a space

    == example of "_word.txt"==
    What kind of memory ?
    We respectfully invite you to watch a special edition of Across China .
    == the end of the example ==

    In each "_word.txt" file contains one sentence's tag per line, tags are seperated by a space

    == example of "_tag.txt"==
    O O O O O
    O O O O O O O O O O B-ORG I-ORG O
    == the end of the example
    '''

    def get_data(self,set_name):
        '''
        Load data from the conll-2012 format files.
        Args: 
            set_name: the folders name for training, testing and validation purpose. 
            In this program, the set_name should be "train", "test" or "development"
        Returns:
            A dataFrame contains the data from the correspoding folder
            This function also create a .csv file in the "\data\csv\" folder
        '''
        current_path = os.getcwd()

        root_path = current_path + "\\data\\conll-2012\\v4\\data\\"+ set_name +"\\data\\english"
        table = []

        # Wolk throw the folders and load data from files endwith "gold_conll"
        for root, subFolders, files in os.walk(root_path):
            files = [fi for fi in files if fi.endswith("gold_conll")]
            for name in files:
                file_path = root + "\\" + name
                file = open(file_path,"r",encoding = 'utf-8')
                data =file.readlines()
                sentences_number = 0
                for line in data:
                    if line.startswith('#begin'):   #eliminate the line indicate the begining position of a document
                        sentences_number =0
                        continue
                    if line.startswith('#end'):     #eliminate the line indicate the ending position of a document
                        continue
                    if not line.strip():            #If a line contains nothing, that's means a new sentence begins
                        sentences_number += 1
                        continue
                    features = re.split('\s+',line.strip())
                    features.insert(2,sentences_number)
                    table.append(features)
        
        columns = ['DocID','PartNum','SentenceNum','WordNum','Word','POS','Parse','Lemma','PFID','Sense','Speaker','NE' ]
        data = pd.DataFrame(table).iloc[:,0:12]
        data.columns = columns
        print(set_name + " data get!")
        data = self.tagging(data)
        data.to_csv(current_path + "\\data\\csv\\"+ set_name +".csv")
        return data

    def tagging(self, data):
        '''
        re-tagging the NE column with "IOB tag-NE tag" format
        arg:
            data: a dataframe contains OntoNote 5.0 data with conll-2012 format
        return:
            a dataframe which the "NE" column is ""IOB tag-NE tag" format"
        '''
        IOB = 'O'
        NE = ''
        for i,row in data.iterrows():
            #print(i)
            if row['NE'] == "*":
                data.at[i,'NE'] = IOB + NE
            elif row['NE'].startswith("("):
                NE = row['NE'][1:-1]
                data.at[i,'NE'] = "B-" + NE
                if row['NE'].endswith(")"):
                    NE = ''
                    continue
                IOB = "I-"
            elif row['NE'].endswith(")"):
                data.at[i,'NE'] = IOB + NE
                IOB = 'O'
                NE = ''
        print("re-tagging the NE part is done!")
        return data

    def rearrange(self, data,name):
        current_path = os.getcwd()
        '''
        extract data from dataFram
        arg:
            data: a dataframe contains OntoNote 5.0 data with conll-2012 format
            name: the name of the dataset, it should be "train", "test" or "development"
        return:
            two encoded lists contains the words and tags
            It also create two txt files named "_word.txt" and "_tag.txt"
        '''
        DocID = None
        PartNum = None
        SentenceNum = None

        words_lines = []
        tag_lines =[]

        words_line =[]
        tag_line =[]

        words_file = open(current_path +"\\data\\text\\"+ name+"_word.txt",'w',encoding='utf-8')
        tag_file = open(current_path +"\\data\\text\\" + name+"_tag.txt",'w',encoding='utf-8')

        for i,row in data.iterrows():
            if DocID != row["DocID"] or PartNum != row["PartNum"] or SentenceNum != row["SentenceNum"]:
                if words_line and tag_line:

                    words = [w.encode(encoding='UTF-8') for w in words_line]
                    words_lines.append(words)
                    tags =[t.encode(encoding='UTF-8') for t in tag_line]
                    tag_lines.append(tags)

                    words_file.write(" ".join(words_line)+"\n")
                    tag_file.write(" ".join(tag_line)+"\n")
                DocID = row["DocID"]
                PartNum = row["PartNum"]
                SentenceNum = row["SentenceNum"]
                words_line = [row["Word"]]
                tag_line = [row["NE"]]
            else:
                words_line.append(row['Word'])
                tag_line.append(row['NE'])
        words_file.close()
        tag_file.close()
        return words_lines, tag_lines

    def load_data(self):
        '''
        try to load data from .txt files, .csv files or conll-2012 format files

        return: encodded lists of words and tags for trainning testing and development data
        '''
        current_path = os.getcwd()
        try:
            train_word = self.read_from_text(current_path + "\\data\\text\\train_word.txt")
            train_tag = self.read_from_text(current_path + "\\data\\text\\train_tag.txt")

            test_word = self.read_from_text(current_path + "\\data\\text\\test_word.txt")
            test_tag = self.read_from_text(current_path + "\\data\\text\\test_tag.txt")

            development_word = self.read_from_text(current_path + "\\data\\text\\development_word.txt")
            development_tag = self.read_from_text(current_path + "\\data\\text\\development_tag.txt")
        except Exception as e:
            print("Can not get txt format files, try to read .csv files")
            try:
                train = pd.read_csv(current_path + "\\data\\csv\\train.csv")
                test =  pd.read_csv(current_path + "\\data\\csv\\test.csv")
                development = pd.read_csv(current_path + "\\data\\csv\\development.csv")
            except:
                print("Can not get .csv formate files, try to read conll-2012 format files. It may takes a while.")
                try:
                    train = self.get_data("train")
                    test =  self.get_data("test")
                    development =  self.get_data("development")
                except Exception as e:
                    print(str(e))
                    return
            train["Word"] = train["Word"].astype(str)
            test["Word"] = test["Word"].astype(str)
            development["Word"] = development["Word"].astype(str)
            train_word, train_tag = self.rearrange(train,"train")
            test_word, test_tag = self.rearrange(test,"test")
            development_word, development_tag = self.rearrange(development,"development")


        return train_word, train_tag,test_word, test_tag,development_word, development_tag

    def read_from_text(slef, path):
        '''
        read data from .txt file
        arg:
            path: the path of the file.
        '''
        file = open(path,'r',encoding='utf-8')
        result = []
        lines =file.readlines()
        for line in lines:
            words = [w.encode(encoding='UTF-8') for w in line.strip().split()]
            result.append(words)
        file.close()
        return result

    def build_voc(self):
        current_path = os.getcwd()
        counter_words = Counter()

        train = self.read_from_text(current_path +"\\data\\text\\train_word.txt")
        test = self.read_from_text(current_path +"\\data\\text\\test_word.txt")
        development = self.read_from_text(current_path +"\\data\\text\\development_word.txt")

        # build word vocabulary
        for dataset in [train,test,development]:
            for line in dataset:
                counter_words.update(line)

        voc_words = {w.decode(encoding = "utf-8") for w,c in counter_words.items() if c >=1}

        file = open(current_path + "\\data\\text\\word_voc.txt",'w',encoding = "utf-8")
        for w in sorted(list(voc_words)):
            file.write(w + "\n")
        file.close()

        # build the character vocabulary
        voc_char = ''
        for w in voc_words:
            voc_char = voc_char + w
        voc_char = set(voc_char)

        file = open(current_path + "\\data\\text\\char_voc.txt",'w',encoding = "utf-8")
        for c in sorted(list(voc_char)):
            file.write(c + "\n")
        file.close()

        # build the tag vocabulary
        train = self.read_from_text(current_path +"\\data\\text\\train_tag.txt")
        test = self.read_from_text(current_path +"\\data\\text\\test_tag.txt")
        development = self.read_from_text(current_path +"\\data\\text\\development_tag.txt")
        counter_tags = Counter()
        for dataset in [train,test,development]:
            for line in dataset:
                counter_tags.update(line)

        voc_tags = {w.decode() for w,c in counter_tags.items() if c >=1}

        file = open(current_path + "\\data\\text\\tag_voc.txt",'w')
        for t in sorted(list(voc_tags)):
            file.write(t + "\n")

    def build_embedding(self):
        current_path = os.getcwd()

        file = open(current_path + "\\data\\text\\word_voc.txt",'r',encoding = "utf-8")
        lines = file.readlines()

        word_id = {line.strip(): id for id, line in enumerate(lines)}
        size = len(word_id)

        embeddings = np.zeros((size, 300))

        file = open(current_path + "\\data\\text\\glove.840B.300d.txt",'r',encoding = "utf-8")
        lines = file.readlines()
        for id,line in enumerate(lines):
            line = line.strip().split()
            if len(line) != 301:
                continue
            word = line[0]
            embedding = line[1:]
            if word in word_id:
                id = word_id[word]
                embeddings[id] = embedding

        np.savez_compressed(current_path + "\\data\\text\\word_embeddings.npz", embeddings=embeddings)


