import os
import numpy as np
import evaluation.evaluation as ie
from evaluation.parser.ibsen_parser import parse_ibsen_file


def test_ibsen():
    #Reference:  '/../measurements/Ostsee/T2/ST06/'
    measurement = os.path.dirname(os.path.realpath(__file__)) + '/data/'

    files = ['reference001.asc', 'target003.asc', 'darkcurrent001.asc']
    file_set  = [measurement + f for f in files]
    print(file_set)
    ie.evaluate(*file_set)

    ref = parse_ibsen_file(file_set[0])
    tar = parse_ibsen_file(file_set[1])
    dark = parse_ibsen_file(file_set[2])

    assert ref['data'][:, 0].all() == ref['tdata'][0].all()

    print("                                                           =================== PASSED =====================")
