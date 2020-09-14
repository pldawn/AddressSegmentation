# -*- coding: utf-8 -*-
import sys
import os
import word2vec
from sentence import Sentence

def processToken(inputchar, sentence, end, vob):
    MAX_LEN = 70
    inputchar = unicode(inputchar)
    sentence.addPredictToken(inputchar)
    uline = u''
    if end:
        if sentence.chars > MAX_LEN:
            pass
        else:
            x = []
            sentence.generate_pr_line(x, vob)
            nn = len(x)
            for j in range(nn, MAX_LEN):
                x.append(0)
            line = ''
            for i in range(MAX_LEN):
                if i > 0:
                    line += " "
                line += str(x[i])
        sentence.clear()
        return line


def processLine(inputline, vob):
    inputline = inputline.strip()
    inputline = inputline.split(' ')
    nn = len(inputline)
    sentence = Sentence()
    for i in range(nn - 1):
        processToken(inputline[i], sentence, False, vob)
    out = processToken(inputline[nn - 1], sentence, True, vob)
    return out
