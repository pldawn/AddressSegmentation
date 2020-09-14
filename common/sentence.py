#!/usr/bin/env python
# -*- coding:utf-8 -*-

# File: sentence.py
# Project: /e/code/kcws
# Created: Thu Jul 27 2017
# Author: Koth Chen
# Copyright (c) 2017 Koth
#
# <<licensetext>>


class Sentence:
    def __init__(self):
        self.tokens = []
        self.chars = 0
        self.schema = ['B', 'I', 'E', 'S', 'O']
        self.entities = ['COUNTRY', 'PROVINCE', 'CITY', 'DISTRICT', 'COUNTY', 'TOWN', 'TOWNSHIP', 'BLOCK', 'VILLAGE', 'REGIONNUM', 'STREET', 'STREETNUM', 'LANDMARK', 'BUILDING', 'FLOOR', 'TABLET']
        self.labelmap = self.get_labelmap(self.schema, self.entities)

    def get_labelmap(self, schema, entities):
        labelmap = {}
        ind = 1
        for e in entities:
            for s in schema[:-1]:
               labelmap[s + '-' + e] = ind
               ind += 1
        labelmap[schema[-1]] = 0
        return labelmap

    def addTrainToken(self, char, label):
        self.chars += len(char)
        self.tokens.append((char, label))

    def addPredictToken(self, char):
        self.chars += len(char)
        self.tokens.append(char)

    def clear(self):
        self.tokens = []
        self.chars = 0
    
    def get_char_index(self, char, model):
        try:
            index = model.ix(char.decode('utf-8'))
            return index
        except KeyError:
            return model.ix('<UNK>') # UNK's index of embedding model is index of <UNK>, padding's index of embedding model is 0

    def generate_tr_line(self, x, y, vob):
        for char, label in self.tokens:
            x.append(self.get_char_index(str(char.encode('utf-8')), vob))
            y.append(self.labelmap[label])

    def generate_pr_line(self, x, vob):
        for char in self.tokens:
            x.append(self.get_char_index(str(char.encode('utf-8')), vob))

