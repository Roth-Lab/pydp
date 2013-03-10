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
from __future__ import division

from pydp.data import BetaData, BetaParameter, GammaData, GammaParameter
from pydp.rvs import beta_rvs, gamma_rvs

class BaseMeasure(object):
    '''
    Base class for base measures.
    '''
    def random(self):
        '''
        Return a random sample from the base measure.
        '''
        pass
    
class BetaBaseMeasure(BaseMeasure):
    def __init__(self, a, b):
        self.params = BetaParameter(a, b)
    
    def random(self):
        x = beta_rvs(self.params.a, self.params.b)
        
        return BetaData(x)
    
class GammaBaseMeasure(BaseMeasure):
    def __init__(self, a, b):
        self.params = GammaParameter(a, b)
    
    def random(self):
        x = gamma_rvs(self.params.a, self.params.b)
        
        return GammaData(x)
