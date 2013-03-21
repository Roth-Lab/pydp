'''
This file is part of PyDP.

PyDP is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

PyDP is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with PyDP.  If not, see
<http://www.gnu.org/licenses/>.

Created on 2012-09-22

@author: Andrew Roth
'''
from pydp.data import BetaData
from pydp.rvs import beta_rvs
from pydp.densities import log_beta_pdf

class ProposalFunction(object):
    def random(self, params):
        pass

class BaseMeasureProposalFunction(ProposalFunction):
    def __init__(self, base_measure):
        self.base_measure = base_measure
        
    def log_p(self, data, params):
        return self.base_measure.log_p(data)
    
    def random(self, params):
        return self.base_measure.random()

class BetaProposalFunction(ProposalFunction):
    def __init__(self, s):
        self.s = s
        
    def log_p(self, data, params):
        a, b = self._get_standard_params(params)
        
        return log_beta_pdf(data.x, a, b)
    
    def random(self, params):
        a, b = self._get_standard_params(params)
        
        a += 1
        b += 1
        
        return BetaData(beta_rvs(a, b))
    
    def _get_standard_params(self, params):
        s = self.s
        m = params.x
        
        a = s * m
        b = s - a
        
        return a, b
