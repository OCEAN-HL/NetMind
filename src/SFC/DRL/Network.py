import copy

import numpy as np
import scipy.stats as stats


class Vertex:
    def __init__(self, key, computing_resource, node_type):
        self.id = key
        self.computing_resource = computing_resource
        self.connected_node = {}  # key = [bandwidth, distance]
        self.distance = 0
        self.node_type = node_type  

    def __str__(self):
        return (
            "Node"
            + str(self.id)
            + " has "
            + str(self.computing_resource)
            + " computing resource, and connected with "
            + "Node "
            + str([x.id for x in self.connected_node])
            + " with distance of "
            + str(x.value for x in self.connected_node)
        )

    def get_id(self):
        return self.id

    def get_resource(self):
        return self.computing_resource

    def add_neighbor(self, nbr, bw, distance):
        if nbr not in self.connected_node:
            self.connected_node[nbr] = [bw, distance]

    def get_bandwidth(self, nbr):
        return self.connected_node[nbr][0]

    def get_distance(self, nbr):
        return self.connected_node[nbr][1]

    def change_computing(self, variation):
        self.computing_resource += variation

    def change_bandwidth(self, nbr, variation):
        self.connected_node[nbr][0] += variation

    def get_neighbors(self):
        return self.connected_node.keys()


class Network:
    def __init__(self):
        self.node_list = {}
        self.total_node_number = 0

    def get_node_list(self):
        return self.node_list.keys()

    def add_node(self, id, computing_resource, node_type):
        self.node_list[id] = Vertex(id, computing_resource, node_type)
        self.total_node_number += 1

    def add_edge(self, node_1, node_2, bw, distance):
        if node_1 in self.node_list and node_2 in self.node_list:
            self.node_list[node_1].add_neighbor(node_2, bw, distance)
            self.node_list[node_2].add_neighbor(node_1, bw, distance)
