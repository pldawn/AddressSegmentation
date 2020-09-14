import os
import sys

class Conditions:
  def __init__(self, conditions_dataset_path):
    self.cond_dict = {}
    with open(conditions_dataset_path, 'r') as f:
      for line in f:
        line = line.decode('utf-8').strip().split(':')
        key = line[0].strip().replace(' ', '')
        value = line[1].strip().split(' ')
        self.cond_dict[key] = value
    self.keys = self.cond_dict.keys()
    self.keys = sorted(self.keys, key=lambda x: len(x))
    self.schema = ['B', 'I', 'E', 'S', 'O']
    self.entities = ['PROVINCE', 'CITY', 'DISTRICT', 'DISTRICTNUM', 'STREET', 'STREETNUM', 'LANDMARK', 'BUILDING', 'FLOOR', 'TABLET']
    self.labelmap = self.get_labelmap(self.schema, self.entities)


  def get_labelmap(self, schema, entities):
    labelmap = {}
    ind = 0
    for e in entities:
      for s in schema[:-1]:
        labelmap[s + '-' + e] = ind
        ind += 1
    labelmap[schema[-1]] = ind
    return labelmap


  def convert_labels(self, labels):
    res = [self.labelmap[label] for label in labels] 
    return res


  def get_single_conditions(self, x):
    if ' 'in x:
      x = x.replace(' ','')
    cond = [-1 for i in x]
    for key in self.keys:
      if key in x:
        start = x.index(key)
        end = start + len(key)
        cond[start: end] = self.convert_labels(self.cond_dict[key])
    return cond

