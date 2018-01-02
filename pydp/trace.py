'''
This file is part of PyDP.

PyDP is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

PyDP is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with PyDP.  If not, see
<http://www.gnu.org/licenses/>.

Created on 2012-05-10

@author: Andrew
'''
import bz2
import csv
import os


class Trace(object):

    def update(self, state):
        raise NotImplementedError


class DiskTrace(object):

    def __init__(self, trace_dir, params, column_names=None, file_name_map=None):
        self.trace_dir = trace_dir

        self.trace_files = {}

        self.params = params

        self.column_names = column_names

        if not os.path.exists(trace_dir):
            os.makedirs(trace_dir)

        for param_name in self.params:
            if file_name_map is not None and param_name in file_name_map:
                self.trace_files[param_name] = os.path.join(trace_dir, "{0}.tsv.bz2".format(file_name_map[param_name]))
            else:
                self.trace_files[param_name] = os.path.join(trace_dir, "{0}.tsv.bz2".format(param_name))

        self._fhs = {}

        self._writers = {}

    def close(self):
        for param_name in self.params:
            self._fhs[param_name].close()

        self._fhs = {}

        self._writers = {}

    def open(self, mode='r'):
        for param_name in self.params:
            self._fhs[param_name] = bz2.BZ2File(self.trace_files[param_name], mode)

            if mode == 'w':
                self._writers[param_name] = csv.writer(self._fhs[param_name], delimiter='\t')

                if self.column_names is not None and param_name != 'alpha':
                    self._writers[param_name].writerow(self.column_names)
            else:
                raise Exception('Only writing to the trace object is currently supported.')

    def update(self, state):
        for param_name in self.params:
            if param_name == 'alpha':
                row = [state['alpha'], ]

            elif param_name == 'labels':
                row = state['labels']

            else:
                row = [getattr(x, param_name) for x in state['params']]

            self._writers[param_name].writerow(row)


class MemoryTrace(Trace):

    def __init__(self):
        self.alpha = []

        self.labels = []

        self.params = []

    def update(self, state):
        self.alpha.append(state['alpha'])

        self.labels.append(state['labels'])

        self.params.append(state['params'])
