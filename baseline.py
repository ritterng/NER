import numpy as np
import pandas as pd
from DataPreProcessing import data_loader as dl
import re
import os

def get_transition_matrix(train_tag):
    transition_matrix = {}
    transition_count = {}

    for line in train_tag:
        tags = [b'O'] + line + [b'O']
        if not line:
            continue
        for n in range(len(tags)):
            tag = tags[n]
            if n+1 < len(tags):
                tag_next = tags[n+1]
            else:
                continue
            if tag not in transition_count:
                transition_matrix[tag] = {}
                transition_count[tag] = {}
            if tag_next not in transition_count[tag]:
                transition_count[tag][tag_next] = 1
            else:
                transition_count[tag][tag_next] += 1

    for tag in transition_count.keys():
        for tag_next in transition_count[tag].keys():
            transition_matrix[tag][tag_next] = transition_count[tag][tag_next]/sum(transition_count[tag].values())
    return transition_matrix


def get_emission_matrix(train_word, train_tag):
    emission_matrix = {}
    emission_count = {}
    
    for word_line, tag_line in zip(train_word,train_tag):
        words = [b'-Start-'] + word_line + [b'-End-']
        tags = [b'O'] + tag_line + [b'O']

        for word,tag in zip(words,tags):
            if tag not in emission_count:
                emission_count[tag] = {}
                emission_matrix[tag] = {}
            if word not in emission_count[tag]:
                emission_count[tag][word] = 1
            else:
                emission_count[tag][word] += 1
    for tag in emission_count.keys():
        for word in emission_count[tag].keys():
            emission_matrix[tag][word] = emission_count[tag][word]/sum(emission_count[tag].values())
    return emission_matrix

def beam(data, b):
    max_result = [-1 for i in range(b)]
    keys =[]
    results = {}
    for key in data.keys():
        if data[key] > min(max_result):
            del max_result[max_result.index(min(max_result))]
            if len(keys) > b-1:
                del keys[max_result.index(min(max_result))]
            max_result.append(data[key])
            keys.append(key)
    for i in range(b):
        results[keys[i]] = max_result[i]
    return results

def viterbi(test_data,emission_matrix,transition_matrix,dict, beambool = False):
    sequence = {}
    prob =[{}]
    words =[b'-Start-'] + test_data + [b'-End-']


    prob[0][b'O'] = 1
    sequence[b'O'] =[b'O']

    for n in range(1,len(words)-1):
        word = words[n]

        prob.append({})
        sequence_rest ={}
        for tag in emission_matrix.keys():
            result =[]
            for tag_before in prob[n-1].keys():
                if word not in dict:
                    result.append((0.000000000000000000000000001,tag_before))
                else:
                    if tag in transition_matrix[tag_before].keys() and word in emission_matrix[tag].keys():
                        result.append((prob[n-1][tag_before] + transition_matrix[tag_before][tag] + emission_matrix[tag][word],tag_before))
                    else:
                        result.append((0,tag_before))
            p,s =max(result)
            sequence_rest[tag] =sequence[s] + [tag]
            prob[n][tag] = p
        if beambool == True:
            prob[n] = beam(prob[n],10)
        sequence =sequence_rest

    p,s = max([(prob[-1][tag_before],tag_before) for tag_before in prob[-1].keys()])
    return sequence[s][1:]

def prediction():
    current_path = os.getcwd()

    a = dl()
    train_word, train_tag,test_word, test_tag,development_word, development_tag  = a.load_data()
    if not os.path.isfile(os.getcwd()+"\\data\\text\\char_voc.txt") and not os.path.isfile(os.getcwd()+"\\data\\text\\tag_voc.txt") and not os.path.isfile(os.getcwd()+"\\data\\text\\word_voc.txt"):
        a.build_voc()
    if not os.path.isfile(os.getcwd()+"\\data\\text\\word_embeddings.npz"):
        a.build_embedding()

    train_word = [x for x in train_word]
    word_voc_file = open(os.getcwd()+"\\data\\text\\word_voc.txt", 'rb')
    word_voc = word_voc_file.readlines()
    word_voc = [re.sub('\r\n','',x.decode('utf-8')).encode('utf-8') for x in word_voc]
    word_voc_file.close()

    transition_matrix = get_transition_matrix(train_tag)
    emiission_matrix = get_emission_matrix(train_word,train_tag)
    predicted = []

    for lines in test_word:
        result = viterbi(lines,emiission_matrix,transition_matrix,word_voc,True)
        predicted.append(result)

    result_path = current_path + "\\results\\baseline.result.txt"
    file = open(result_path,'wb')

    for words, grounds, predicteds in zip(test_word, test_tag, predicted):
        for word, tags, tag_pred in zip(words, grounds, predicteds):
            file.write(b' '.join([word,tags,tag_pred]) + b'\n')
            file.write(b'\n')

    file.close()

if __name__ == "__main__":
    prediction()






