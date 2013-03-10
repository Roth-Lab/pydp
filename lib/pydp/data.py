'''
This file is part of PyDP.

PyDP is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

PyDP is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with PyDP.  If not, see
<http://www.gnu.org/licenses/>.

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
