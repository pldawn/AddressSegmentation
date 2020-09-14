# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import tensorflow as tf
import word2vec as w2v

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('predict_mode', 2, '0: crf frozen graph, 1: viterbi frozen graph, 2: normal frozen graph')
tf.app.flags.DEFINE_string('inputs', '中关村大街E世界大厦', 'test inputs')
tf.app.flags.DEFINE_string('script_dir', os.path.dirname(os.path.realpath(__file__)), 'main script direcoty')
tf.app.flags.DEFINE_string('common_dir', FLAGS.script_dir + '/common', 'relative packages directory')
tf.app.flags.DEFINE_string('lib_dir', FLAGS.script_dir + '/lib', 'embedding and pre-specifing info directory')
tf.app.flags.DEFINE_string('model_dir', FLAGS.script_dir + '/model', 'frozen checkpoint directory')
tf.app.flags.DEFINE_string('vectors_path', FLAGS.lib_dir + '/vec.txt', 'pretrained word vectors file path')
tf.app.flags.DEFINE_string('conditions_path', FLAGS.lib_dir + '/prior_conditions.txt', 'pre-specifing dict')
tf.app.flags.DEFINE_string('crf_frozen_graph', FLAGS.model_dir + '/crf_fronzen_ckpt.pb', 'crf frozen graph name')
tf.app.flags.DEFINE_string('viterbi_frozen_graph', FLAGS.model_dir + '/viterbi_frozen_ckpt.pb', 'viterbi frozen graph name')
tf.app.flags.DEFINE_string('normal_frozen_graph', FLAGS.model_dir + '/addrNer.pb', 'normal frozen graph name')

sys.path.append(FLAGS.common_dir)

import generate_prediction
import modify_conditions
import crf_frozen_graph
import viterbi_frozen_graph
import tag_merge


def segment(sent):
  if not sent:
    return ''
  else:
    tmp = []
    for c in sent:
      tmp.append(c)
    return ' '.join(tmp)


def main(argv):
  inputs = FLAGS.inputs.strip()
  
  # segment
  pre_inputs = segment(inputs.decode('utf-8'))
  input_len = [len(pre_inputs.split(' '))]                     # shape = [?], ? = test_batch_size

  # map to vector index
  vob = w2v.load(FLAGS.vectors_path)
  features = generate_prediction.processLine(pre_inputs, vob)  # string
  features = np.array(features.split(' '), np.int32)
  features = np.expand_dims(features, 0)                       # shape = [1,70], dtype = int32
  
  # get prior label conditions
  if FLAGS.predict_mode in (0, 1):
    condset = modify_conditions.Conditions(FLAGS.conditions_path)
    conditions = condset.get_single_conditions(pre_inputs)       # shape = [pre_inputs_length]
    conditions = np.array(conditions + [-1 for _ in range(features.shape[1] - len(conditions))], np.int32)
    conditions = np.expand_dims(conditions, 0)                   # shape = [1,70], dtype = int32

  # predict
  # feature.shape = [?,70], conditions.shape = [?,70], input_len.shape = [?], ? = test_batch_size
  if FLAGS.predict_mode == 0:
    predictor = crf_frozen_graph.Model()
    predictor.load_graph(FLAGS.crf_frozen_graph)
    with tf.Session(graph=predictor.graph) as sess:
      predictor.load_tensors(sess)
      labels = predictor.predict(sess, features, input_len, conditions)  # string
  
  elif FLAGS.predict_mode == 1:
    predictor = viterbi_frozen_graph.Model()
    predictor.load_graph(FLAGS.viterbi_frozen_graph)
    with tf.Session(graph=predictor.graph) as sess:
      predictor.load_tensors(sess)
      labels = predictor.predict(sess, features, input_len, conditions)  # string

  elif FLAGS.predict_mode == 2:
    predictor = crf_frozen_graph.Model()
    predictor.load_graph(FLAGS.normal_frozen_graph)
    with tf.Session(graph=predictor.graph) as sess:
      predictor.load_tensors(sess)
      labels = predictor.predict(sess, features, input_len)  # string
  
  else:
    print('valid mode: 0: crf frozen graph, 1: viterbi frozen graph, 2: normal frozen graph')
    sys.exit(0)

  # merge labels to entities
  merge = tag_merge.Merge()
  output = merge.mergeline(pre_inputs, labels)                 # string
  # print(output.encode('utf-8'))
  return output


if __name__ == '__main__':
  tf.app.run()

