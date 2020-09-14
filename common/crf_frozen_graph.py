import os
import sys
import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Model:
  def __init__(self):
    self.prefix = 'recognizer'
    self.crf_name = 'crf_sequences:0'
    self.crf = None
    self.input_holder_name = 'input_placeholder:0'
    self.input_holder = None
    self.cond_holder_name = 'conditions_placeholder:0'
    self.cond_holder = None
    self.schema = ['B', 'I', 'E', 'S', 'O']
    self.entities = ['COUNTRY', 'PROVINCE', 'CITY', 'DISTRICT', 'COUNTY', 'TOWN', 'TOWNSHIP', 'BLOCK', 'VILLAGE', 'REGIONNUM', 'STREET', 'STREETNUM', 'LANDMARK', 'BUILDING', 'FLOOR', 'TABLET']
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
    try:
      self.cond_holder = self.graph.get_tensor_by_name(self.prefix + '/' + self.cond_holder_name)
    except:
      pass
    # print('succeed to load conditions placeholder')
    self.crf = self.graph.get_tensor_by_name(self.prefix + '/' + self.crf_name)
    # print('succeed to load sequences')
    # print(self.input_holder,self.cond_holder,self.crf)
  

  def predict(self, sess, x, x_len, conditions=None):
    '''
    x.shape = [?, 70], conditions.shape = [?, 70], x_len.shape = [?], ? = batch
    '''
    feed_dict = {self.input_holder: x}
    if conditions is not None:
      feed_dict[self.cond_holder] =  conditions
    # crf decode
    sequences= sess.run(self.crf, feed_dict=feed_dict)
    tmp = []
    for sequence, lth in zip(sequences, x_len):
      valid_sequence = sequence[:lth]
      for label in valid_sequence:
        true_label = self.labelmap_reverse[label]
        tmp.append(true_label)
    output = ' '.join(tmp)
    return output  
        

def main(argv):
  model = Model()
  model.load_graph(argv[1])

  with tf.Session(graph=model.graph) as sess:
    model.load_tensors(sess)
    output = model.predict(sess, argv[2], argv[3], argv[4])
  return output
