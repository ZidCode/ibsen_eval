import theano
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Model import IrradianceModel_sym
from FitModel import FitWrapper
from Residuum import Residuum
from get_ssa import get_ssa
from scipy.optimize import minimize, least_squares
from Sensitivity_Analysis import start



if __name__ == "__main__":
    import argparse
    import logging
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--level', default='DEBUG')
    args = parser.parse_args()
    logger = logging.getLogger('sensivity_analysis')
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel(args.level)
    logger.info("Start") 
    start(logger)
