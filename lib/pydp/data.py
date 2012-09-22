'''
Created on 2012-09-21

@author: Andrew Roth
'''
from collections import namedtuple

BetaData = namedtuple('BetaData', 'x')

BetaParameter = namedtuple('BetaPriorData', ['a', 'b'])

BinomialData = namedtuple('BinomialData', ['x', 'n'])

GammaData = namedtuple('GammaData', 'x')

GammaParameter = namedtuple('GammaParameter', ['a', 'b'])

PoissonData = namedtuple('PoissonData', 'x')

NegativeBinomialParameter = namedtuple('NegativeBinomialParameter', ['r', 'p'])
