'''
Created on 2012-09-21

@author: Andrew Roth
'''
from collections import namedtuple

BetaData = namedtuple('BetaData', 'x')

BetaPriorData = namedtuple('BetaPriorData', ['a', 'b'])

BinomialData = namedtuple('BinomialData', ['x', 'n'])
