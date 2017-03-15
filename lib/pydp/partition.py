'''
This file is part of PyDP.

PyDP is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

PyDP is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with PyDP.  If not, see
<http://www.gnu.org/licenses/>.

Created on 2012-09-20

@author: Andrew Roth
'''


class Partition(object):

    def __init__(self):
        self.cells = []

    @property
    def cell_values(self):
        return [cell.value for cell in self.cells]

    @property
    def counts(self):
        return [cell.size for cell in self.cells]

    @property
    def item_values(self):
        cell_values = self.cell_values
        labels = self.labels

        return [cell_values[i] for i in labels]

    @property
    def labels(self):
        labels = [None] * self.number_of_items

        for cell_index, cell in enumerate(self.cells):
            for item in cell.items:
                labels[item] = cell_index

        return labels

    @property
    def number_of_cells(self):
        return len(self.cells)

    @property
    def number_of_items(self):
        return sum(self.counts)

    def add_cell(self, value):
        cell = PartitionCell(value)

        self.cells.append(cell)

        return cell

    def add_item(self, item, cell_index):
        self.cells[cell_index].add_item(item)

    def get_cell_by_value(self, value):
        for cell in self.cells:
            if cell.value == value:
                return cell

    def remove_item(self, item, cell_index):
        self.cells[cell_index].remove_item(item)

    def remove_empty_cells(self):
        for cell in self.cells[:]:
            if cell.empty:
                self.cells.remove(cell)

    def copy(self):
        partition = Partition()

        for cell_index, cell in enumerate(self.cells):
            partition.add_cell(cell.value)

            for item in cell.items:
                partition.add_item(item, cell_index)

        return partition


class PartitionCell(object):

    def __init__(self, value):
        self.value = value

        self._items = []

    @property
    def empty(self):
        if self.size == 0:
            return True
        else:
            return False

    @property
    def items(self):
        return self._items[:]

    @property
    def size(self):
        return len(self._items)

    def add_item(self, item):
        self._items.append(item)

    def remove_item(self, item):
        self._items.remove(item)

    def __contains__(self, x):
        if x in self._items:
            return True
        else:
            return False
