import os
import numpy as np
import evaluation.evaluation as ie
from evaluation.parser.ibsen_parser import parse_ibsen_file


def test_evaluate():
    measurement = os.path.dirname(os.path.realpath(__file__)) + '/../../measurements/Ostsee/T2/ST06/'
    DEBUG=False
    files = ['reference001.asc', 'target003.asc', 'darkcurrent001.asc']
    file_set  = [measurement + f for f in files]
    file_set.append(DEBUG)
    ie.evaluate(*file_set)

