import theano
import numpy as np
from theano import tensor as T
from scipy.constants import atmosphere


to_list = lambda x : [x]


class Residuum:

    def __init__(self, model, name):
        self.model = model.getModel(name)
        self.symbols = model.get_Symbols()
        self.reference = T.vector('reference')
        self.args = to_list(self.reference) + self.symbols

    def getsymResiduum(self):
        R = 0.5* T.sum((self.model - self.reference)**2)
        return R

    def getResiduum(self):
        R = self.getsymResiduum()
        f = theano.function(self.args, R)
        return f

    def getResiduals(self):
        R = self.model - self.reference
        f = theano.function(self.args, R)
        return f

    def getDerivative(self):
        R = self.getsymResiduum()
        dR = T.grad(R, self.symbols)
        f = theano.function(self.args, dR)
        return f

