import pandas as pd
import numpy as np
from DataPreProcessing import data_loader as dl
import tensorflow as tf
import functools
import os
import logging
import sys
import warnings

warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.logging.set_verbosity(logging.INFO)
handlers = [logging.FileHandler(os.getcwd() +'\\results\\main.log'),logging.StreamHandler(sys.stdout)]
logging.getLogger('tensorflow').handlers = handlers

def generator_fn(train_word, train_tag):
    for w,t in zip(train_word, train_tag):
        assert len(w) == len(t)
        yield (w,len(w)),t

def input_fn(words, tags, params=None, shuffle_and_repeat=False):
     params = params if params is not None else {}
     shapes = (([None], ()), [None])
     types = ((tf.string, tf.int32), tf.string)
     defaults = (('<pad>', 0), 'O')

     dataset = tf.data.Dataset.from_generator(functools.partial(generator_fn, words, tags),output_shapes=shapes, output_types=types)

     if shuffle_and_repeat:
         dataset = dataset.shuffle(params['buffer']).repeat(params['epochs'])

     dataset = (dataset
                .padded_batch(params.get('batch_size', 300), shapes, defaults)
                .prefetch(1))
     return dataset

def model_fn(features, labels, mode, params):

     #load datas
     training = (mode == tf.estimator.ModeKeys.TRAIN)
     words,num = features
     voc_word = tf.contrib.lookup.index_table_from_file(params['words'], num_oov_buckets = params['num_oov_buckets'])
     voc_tag = read_file(params['tags'])
     index = [id for id, tag in enumerate(voc_tag) if tag.strip()!='O']
     num_tags = len(index) + 1

     #word Embeddings
     word_ids = voc_word.lookup(words)
     embeded = np.load(params['embeded'])['embeddings']
     variable = np.vstack([embeded,[[0.]*params['dim']]])
     variable = tf.Variable(variable,dtype=tf.float32,trainable = False)
     embeddings = tf.nn.embedding_lookup(variable,word_ids)
     embeddings = tf.layers.dropout(embeddings,rate = params['dropout'],training = training)

     #LSTM
     t = tf.transpose(embeddings, perm = [1,0,2])
     lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
     lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
     lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
     output_fw, _ = lstm_cell_fw(t, dtype=tf.float32, sequence_length=num)
     output_bw, _ = lstm_cell_bw(t, dtype=tf.float32, sequence_length=num)
     output = tf.concat([output_fw, output_bw], axis=-1)
     output = tf.transpose(output, perm=[1, 0, 2])
     output = tf.layers.dropout(output, rate=params['dropout'], training=training)

     #CRF
     logits = tf.layers.dense(output, num_tags)
     crf_params = tf.get_variable("crf", [num_tags, num_tags], dtype=tf.float32)
     pred_ids, _ = tf.contrib.crf.crf_decode(logits, crf_params, num)

     if mode == tf.estimator.ModeKeys.PREDICT:
         reverse_voc_tags = tf.contrib.lookup.index_to_string_table_from_file(params['tags'])
         pred_strings = reverse_voc_tags.lookup(tf.to_int64(pred_ids))
         predctions = {
             'pred_ids': pred_ids, 
             'tag': pred_strings
             }
         return tf.estimator.EstimatorSpec(mode, predictions = predctions)
     else:
         voc_tags = tf.contrib.lookup.index_table_from_file(params['tags'])
         tags = voc_tags.lookup(labels)
         log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
             logits, tags, num, crf_params)
         loss = tf.reduce_mean(-log_likelihood)

         # Metrics
         weights = tf.sequence_mask(num)
         metrics = {
             'accuracy': tf.metrics.accuracy(tags, pred_ids, weights),
         }
         for metric_name, op in metrics.items():
             tf.summary.scalar(metric_name, op[1])

         if mode == tf.estimator.ModeKeys.EVAL:
             return tf.estimator.EstimatorSpec(
                 mode, loss=loss, eval_metric_ops=metrics)

         elif mode == tf.estimator.ModeKeys.TRAIN:
             train_op = tf.train.AdamOptimizer().minimize(
                 loss, global_step=tf.train.get_or_create_global_step())
             return tf.estimator.EstimatorSpec(
                 mode, loss=loss, train_op=train_op)


def read_file(path):
    file = open(path, 'r', encoding = "utf-8")
    data =file.readlines()
    file.close
    return data


if __name__ == '__main__':

    current_path = os.getcwd()

    #load and pre-processing data
    a = dl()
    train_word, train_tag,test_word, test_tag,development_word, development_tag = a.load_data()
    if not os.path.isfile(os.getcwd()+"\\data\\text\\char_voc.txt") and not os.path.isfile(os.getcwd()+"\\data\\text\\tag_voc.txt") and not os.path.isfile(os.getcwd()+"\\data\\text\\word_voc.txt"):
        a.build_voc()
    if not os.path.isfile(os.getcwd()+"\\data\\text\\word_embeddings.npz"):
        a.build_embedding()

    #predictions
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
    train_data = functools.partial(input_fn,train_word,train_tag,params,shuffle_and_repeat = True)
    deve_data = functools.partial(input_fn,development_word,development_tag)

    cfg = tf.estimator.RunConfig(save_checkpoints_secs=120)
    estimator = tf.estimator.Estimator(model_fn, current_path + "\\results\\model", cfg, params)
    hook = tf.contrib.estimator.stop_if_no_increase_hook(estimator, 'accuracy', 500, min_steps=8000, run_every_secs=120)
    train_spec = tf.estimator.TrainSpec(input_fn=train_data, hooks=[hook])
    eval_spec = tf.estimator.EvalSpec(input_fn=deve_data, throttle_secs=120)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
  
    def prediction():
        path = current_path + "\\results\\test.result.txt"
        file = open(path,'wb')
        
        test_data = functools.partial(input_fn,test_word,test_tag)
        ground = generator_fn(test_word,test_tag)
        predicted = estimator.predict(test_data)
        for grou, pred in zip(ground,predicted):
            ((words,_),tags) =grou
            for word, tags, tag_pred in zip(words, tags, pred['tag']):
                file.write(b' '.join([word,tags,tag_pred]) + b'\n')
                file.write(b'\n')

        file.close()

    prediction()

    #word-level accuracy
    def accuracy(path):
        count = 0
        correct = 0
        file = open(path, 'r', encoding="utf-8")
        data = file.readlines()
        for line in data:
            line = line.replace('\n','')
            if line :
                count += 1
                predicted = line.strip().split()
                if predicted[1] == predicted[2]:
                    correct += 1
        return correct/count
            

