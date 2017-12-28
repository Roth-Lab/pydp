'''
This file is part of PyDP.

PyDP is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

PyDP is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with PyDP.  If not, see
<http://www.gnu.org/licenses/>.

Created on 2012-09-18

@author: andrew
'''
from pydp.partition import Partition
from pydp.rvs import discrete_rvs


def sample_from_crp(alpha, size, base_measure):
    labels = []
    values = []

    tables = []

    # Seat the first customer
    tables.append([0, ])

    labels.append(0)
    values.append(base_measure.random())

    for customer in range(1, size):
        p = _get_table_probabilities(tables, alpha)

        table_id = discrete_rvs(p)

        if table_id == len(tables):
            tables.append([customer, ])

            values.append(base_measure.random())
        else:
            tables[table_id].append(customer)

        labels.append(table_id)

    partition = Partition()

    for v in values:
        partition.add_cell(v)

    for item, cell_index in enumerate(labels):
        partition.add_item(item, cell_index)

    return partition


def _get_table_probabilities(tables, alpha):
    p = []

    for table in tables:
        p.append(len(table))

    p.append(alpha)

    p = [x / sum(p) for x in p]

    return p
