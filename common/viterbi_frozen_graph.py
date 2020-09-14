import sys
import tensorflow as tf
import numpy as np
import pickle


class Model:
  def __init__(self):
    self.prefix = 'recognizer'
    self.positions_name = 'Reshape_7:0'
    self.positions = None
    self.transitions_name = 'transitions:0'
    self.transitions = None
    self.input_holder_name = 'input_placeholder:0'
    self.input_holder = None
    self.schema = ['B', 'I', 'E', 'S', 'O']
    self.entities = ['PROVINCE', 'CITY', 'DISTRICT', 'DISTRICTNUM', 'STREET', 'STREETNUM', 'LANDMARK', 'BUILDING', 'FLOOR', 'TABLET']
    self.labelmap_reverse = self.get_labelmap_reverse(self.schema, self.entities)


  def get_labelmap_reverse(self, schema, entities):
        labelmap = {}
        ind = 1
        for e in entities:
            for s in schema[:-1]:
               labelmap[ind] = s + '-' + e
               ind += 1
        labelmap[0] = schema[-1]
        return labelmap


  def load_graph(self, graphdef_path):
    with tf.gfile.GFile(graphdef_path, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
      tf.import_graph_def(graph_def, name=self.prefix)

    self.graph = graph
    # print('succeed to load graph')
    

  def load_tensors(self, sess):
    self.input_holder = self.graph.get_tensor_by_name(self.prefix + '/' + self.input_holder_name)
    # print('succeed to load input placeholder')
    self.positions = self.graph.get_tensor_by_name(self.prefix + '/' + self.positions_name)
    # print('succeed to load posiiton scores')
    self.transitions = self.graph.get_tensor_by_name(self.prefix + '/' + self.transitions_name)
    # print('succeed to load transition scores')

  
  def predict(self, sess, x, x_len, conditions):
    '''
    x.shape = [?, 70], conditions.shape = [?, 70], x_len.shape = [?], ? = batch
    '''
    feed_dict = {self.input_holder: x}
    # crf viterbi decode with conditions
    positions_val, transitions = sess.run([self.positions, self.transitions], feed_dict=feed_dict)
    positions_, length_, conditions_ = positions_val[0], x_len[0], conditions[0]
    positions_, conditions_ = positions_[:length_], conditions_[:length_]
    labels, labels_score = tf.contrib.crf.viterbi_decode_with_conditions(positions_, transitions, conditions_)
    # output
    tmp = []
    for label in labels:
      true_label = self.labelmap_reverse[label]
      tmp.append(true_label)
      output = ' '.join(tmp)
    return output


def main(argv):
  with tf.Session() as sess:
    model = Model()
    model.load_graph(sess, argv[1], argv[2])
    model.predictfile(sess, argv[3], argv[4], argv[5])
