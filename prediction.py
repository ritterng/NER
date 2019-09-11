import functools
import tensorflow as tf
from NER import model_fn
import os
import nltk
import sys


def input_fn(sentence):

    words = [w.encode("utf-8") for w in sentence.strip().split()]
    nwords = len(words)

    words = tf.constant([words], dtype=tf.string)
    nwords = tf.constant([nwords], dtype=tf.int32)

    return (words,nwords),None



def prediction(sentence):
    current_path = os.getcwd()
    Model = current_path + "\\results\\model"
    params = {
        'dim': 300,
        'dropout': 0.5,
        'num_oov_buckets': 1,
        'epochs': 25,
        'batch_size': 300,
        'buffer': 15000,
        'lstm_size': 100,
        'words': current_path + "\\data\\text\\word_voc.txt",
        'chars': current_path + "\\data\\text\\char_voc.txt",
        'tags': current_path + "\\data\\text\\tag_voc.txt",
        'embeded': current_path + "\\data\\text\\word_embeddings.npz"
    }

    estimator = tf.estimator.Estimator(model_fn, Model, params=params)

    input = functools.partial(input_fn,sentence)
    predicted = estimator.predict(input)

    return([t.decode("utf-8") for t in predicted.__next__()['tag']])

def NE(sentences, NE_list):

    result = []
    for sentence, tags in zip(sentences,NE_list):
        ne = []
        cat = ""
        for word, tag in zip(sentence.strip().split(), tags):
            if tag.startswith('O') and ne:
                result.append(cat +": " + ' '.join(ne))
                ne = []
            if tag.startswith('B'):
                if ne:
                    result.append(cat +": " + ' '.join(ne))
                ne = []
                ne.append(word)
                cat = tag.strip().split("-")[1]
            if tag.startswith('I'):
                if ne:
                    ne.append(word)
    return result

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.ERROR)
    try:
        file_name = sys.argv[1]
    except:
        file_name = "text.txt"
    try:
        path = os.getcwd() + "\\" + file_name
        print("Reading text from " + path)
        file = open(path,'r',encoding = 'utf-8')
        text = file.read()
        file.close()
    except Exception as e:
        print(e)
        print("failed to read file!")
        sys.exit()
    print("\n")
    print("Analyzing...")
    sentences = nltk.sent_tokenize(text)

    tags =[]

    for sent in sentences:
        tags.append(prediction(sent))

    nes = NE(sentences,tags)

    result_path = os.getcwd() +"\\ne." + file_name
    file = open(result_path,'w',encoding = 'utf-8')
    for ne in nes:
        file.write(ne + "\n")
    file.close()

    print("The result has been print to ne." + file_name)
    



