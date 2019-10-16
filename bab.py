"""
16th of October
A Problem of Optimal Location of Given Set of Base Stations in Wireless
Networks with Linear Topology

Branch And Bound Algorithm For Binary Search Tree
"""
from collections import deque
from table import Table
from figure import draw
from input import placement, gateway_placement, sta

import networkx as nx
import numpy as np


class Node:
    """Create node of binary search tree"""
    def __init__(self, pi=None, key=0):
        self.key = key
        self.pi = pi
        self.left = None
        self.right = None
        self.estim = None
        self.not_close = True


class BST:
    """Binary search tree for our optimal placement problem"""
    def __init__(self, place, gateway, station):
        self.place = place
        self.gtw = gateway
        self.sta = station
        self.cov = tuple(self.sta[i]['r'] for i in self.sta)
        self.comm_dist = tuple(self.sta[i]['R'] for i in self.sta)
        self.top = None
        self.graph = nx.DiGraph()
        self.table = Table()
        self.unchecked_node = deque()  # Stack of unchecked right child nodes
        self.nodes = deque()  # Stack of all nodes

    def get_indices(self, pi):
        """
        get station j, that we can place to the placement i
        :param pi: matrix
        :return: get indices of first empty pi
        """
        row, col = np.where(pi == 1)
        min_a = None
        min_s = None
        loop_break = False
        for i in range(len(self.place)):
            if loop_break:
                break
            for j in range(len(self.sta)):
                if (i not in row) and (j not in col):
                    if pi[i, j] != 0:
                        min_a = i
                        min_s = j
                        loop_break = True
                        break
        return [min_a, min_s]

    @staticmethod
    def get_child_pi(i, j, node):
        """get matrix pi for left child node and right child node
        :param node: parent node
        :return: left_pi, right_pi
        """
        left_pi = node.pi.copy()
        left_pi[i, j] = 1
        right_pi = node.pi.copy()
        right_pi[i, j] = 0
        return [left_pi, right_pi]

    def is_able2place(self, node):
        """

        :param node: current node
        :return: True, if it is able to place; False - otherwise
        """
        sum_by_row = node.pi.sum(axis=1)
        forbidden_place = sum_by_row[np.where(sum_by_row == 0)]
        if len(self.place) - len(forbidden_place) < len(self.sta):
            return False
        if len(self.unchecked_node) is 0 and node.not_close is False:
            return False
        return True

    @staticmethod
    def _in_range(place1, place2, communication_distance):
        """check whether 'first' and 'second' position is in 'range_' """
        return abs(place1 - place2) <= communication_distance

    def is_able_exist(self):
        """ checking the existence of feasible solution """
        max_range = max(self.comm_dist)
        without_max_range = tuple(i for i in self.comm_dist if i != max_range)
        max_range_2 = max(without_max_range)

        last = self.gtw[-1]

        if not (self._in_range(0, self.place[0], max_range) and
                self._in_range(last, self.place[-1], max_range)):
            return False

        for i in range(len(self.place) - 1):
            if not self._in_range(self.place[i], self.place[i+1], max_range_2):
                return False

        return True

    def initiate_tree(self):
        """ get init estimate and parent init node """
        pi = np.ones([len(self.place), len(self.sta)]) * np.inf
        key = 0
        self.nodes.append(key)
        self.top = Node(pi, key)
        self.graph.add_node(self.top.key)
        root = self.top
        init_noncoverage = sum([2 * i for i in self.cov])
        self.top.estim = max(self.gtw[-1] - init_noncoverage, 0)
        self.table.add(np.inf, np.inf, np.inf, self.top.estim, key)
        self.table.record.append(self.gtw[-1])
        return [root, key]

    def is_able_link_left(self, node, p_ind, s_ind):
        """

        :param node: current node
        :param p_ind: index of placement
        :param s_ind: index of station
        :return: -True if communication distance is more than the distance to
         the left station, -False otherwise
        """
        current_place = self.place[p_ind]
        current_comm_dist = self.comm_dist[s_ind]
        if 1 in node.pi:
            i, j = np.where(node.pi == 1)
            left_place = self.place[i[-1]]
            left_comm_dist = self.comm_dist[j[-1]]
            if not self._in_range(left_place, current_place, left_comm_dist):
                node.not_close = False
                return node.not_close
        else:
            left_place = self.gtw[0]
        if not self._in_range(left_place, current_place, current_comm_dist):
            node.not_close = False
            return node.not_close
        return node.not_close

    def is_able_link_right(self, node, p_ind, s_ind):
        """
        check right of placement point
        :param node: current node
        :param p_ind: index of placement
        :param s_ind: index of station
        :return: (bool) -True if link range is greater than distance or
            False otherwise
        """
        current_place = self.place[p_ind]
        current_comm_dist = self.comm_dist[s_ind]
        _, j = np.where(node.pi == 1)
        placed_comm_dist = [self.comm_dist[i] for i in j] + [current_comm_dist]

        if current_place == self.place[-1] or \
           len(placed_comm_dist) == len(self.comm_dist):
            right_place = self.gtw[-1]
        else:
            right_place = self.place[p_ind+1]
            unplaced_comm_dist = [self.comm_dist[i]
                                  for i in range(len(self.comm_dist))
                                  if (i not in j) and i != s_ind]
            max_comm_dist = max(unplaced_comm_dist)
            if not self._in_range(right_place, current_place, max_comm_dist):
                node.not_close = False
                return node.not_close

        if not self._in_range(right_place, current_place, current_comm_dist):
            node.not_close = False
            return node.not_close

        return node.not_close

    def is_able_within_unplaced(self, node, p_ind, s_ind):
        """
        check unplaced station and right places
        :param node: current node
        :param p_ind: index of placement
        :param s_ind: index of station
        :return: -True if link range of unplaced station is greater than
            distance or False otherwise
        """
        _, j = np.where(node.pi == 1)
        unplaced = [self.comm_dist[i]
                    for i in range(len(self.comm_dist))
                    if (i not in j) and i != s_ind]

        if len(unplaced) == 1:
            node.not_close = False
            for right_p in (self.place[p_ind+1:]):
                node.not_close = self._in_range(self.place[p_ind],
                                                right_p,
                                                unplaced[0]) and\
                                  self._in_range(right_p,
                                                 self.gtw[-1],
                                                 unplaced[0])
                if node.not_close:
                    return node.not_close
            return node.not_close

        if len(unplaced) > 1:
            max_range = max(unplaced)
            unplaced.remove(max_range)
            max_range_2 = max(unplaced)

            for i in range(len(self.place[p_ind+1:])):
                if i == len(self.place) - 1:
                    if not self._in_range(self.gtw[-1],
                                          self.place[i],
                                          max_range):
                        node.not_close = False
                        return node.not_close
                else:
                    if not self._in_range(self.place[i+1],
                                          self.place[i],
                                          max_range_2):
                        node.not_close = False
                        return node.not_close
        return node.not_close

    def get_placement(self, pi):
        """

        :param pi: pi matrix
        :return: to print placed stations
        """
        i, j = np.where(pi == 1)
        placed_sta = [np.inf] * len(self.place)
        for k in range(len(i)):
            placed_sta[i[k]] = j[k] + 1
        print('Placed stations = ', placed_sta)

    def check_estimate(self, node):
        """
        check able to close a node and add new record
        :param node:
        :return: True or False
        """

        if not node.estim < self.table.record[-1]:
            node.not_close = False
        else:
            i, _ = np.where(node.pi == 1)
            if len(i) == len(self.sta):
                self.table.record.append(node.estim)

                self.get_placement(node.pi)
                node.not_close = False

        return node.not_close

    def check_link(self, node, i, j):
        """
        Check link conditions
        :param node: current node
        :param i: index of placement
        :param j: index of station
        :return: True or False
        """
        if self.is_able_link_left(node, i, j) and \
                self.is_able_link_right(node, i, j) and \
                self.is_able_within_unplaced(node, i, j):
            return True
        return False

    @staticmethod
    def noncov_inrange(place1, place2, cov1, cov2):
        """
        Calculate estimate of noncoverage in range between place1 and place2
        :param place1:
        :param place2:
        :param cov1:
        :param cov2:
        :return: Noncoverage between two placed station
        """
        dist = abs(place2 - place1)
        cov = cov1 + cov2
        return max([dist - cov, 0])

    def noncoverage(self, p_ind, s_ind, parent):
        """
        calculate the estimates of noncoverage
        :param p_ind: index of placement
        :param s_ind: index of station
        :param parent: parent node
        :return: Noncoverage estimate
        """
        i, j = np.where(parent.pi == 1)
        if len(i) is 0:
            place1 = self.gtw[0]
            cov1 = 0
        else:
            place1 = self.place[i[-1]]
            cov1 = self.cov[j[-1]]
        # left noncoverage
        left_noncov = parent.estim + self.noncov_inrange(place1,
                                                         self.place[p_ind],
                                                         cov1,
                                                         self.cov[s_ind])
        # right noncoverage
        unplaced_cov = [self.cov[i] for i in range(len(self.cov))
                           if (i not in j) and (i != s_ind)]

        right_noncov = self.noncov_inrange(self.place[p_ind],
                                           self.gtw[-1],
                                           self.cov[s_ind],
                                           sum(2*unplaced_cov))

        return left_noncov + right_noncov

    def add_child_nodes(self, node, key):
        """
        To add new child nodes
        :param node: parent node
        :param key: node number
        :return:
        """
        i, j = self.get_indices(node.pi)
        if self.check_estimate(node) and \
                [i, j] != [None, None]:
            left_pi, right_pi = self.get_child_pi(i, j, node)

            # add left node
            key += 1
            self.nodes.append(key)
            node.left = Node(left_pi, key)
            node.left.estim = self.noncoverage(i, j, node)
            self.graph.add_edge(node.key, node.left.key)
            self.table.add(i, j, node.left.pi[i, j], node.left.estim, key)
            # draw(self.graph)

            # add right node
            key += 1
            self.nodes.append(key)
            node.right = Node(right_pi, key)
            node.right.estim = node.estim
            self.graph.add_edge(node.key, node.right.key)
            self.table.add(i, j, node.left.pi[i, j], node.right.estim, key)

            # draw(self.graph)
            self.unchecked_node.append(node.right)
            # return [node.left, key]
            if self.check_link(node, i, j):
                return [node.left, key]
            return [node.right, key]
        else:
            if len(self.unchecked_node) is not 0:
                node = self.unchecked_node[-1]
                self.unchecked_node.pop()

        return [node, self.nodes[-1]]

    def search(self):
        """
        Search binary tree solution
        :return: record of non-coverage
        """
        assert self.is_able_exist(), 'There is not solution'
        [parent, key] = self.initiate_tree()
        while self.is_able2place(parent):
            parent, key = self.add_child_nodes(parent, key)


solution = BST(placement, gateway_placement, sta)
solution.search()
# draw(solution.graph)
print('record =  ', solution.table.record[-1])
