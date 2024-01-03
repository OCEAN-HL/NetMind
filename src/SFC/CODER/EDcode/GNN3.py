import csv
import json
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_max_pool


# A Class to generate raw data to ML appliable data
class DataProcessing:
    def __init__(self, row_data):
        self.row_data = row_data
        self.train_data_list = []
        self.test_data_list = []

    def Process_data(self, batch_size):
        random.shuffle(self.row_data)
        len_of_total = len(self.row_data)
        split_point = math.ceil(len_of_total * 0.7)
        train_row_data = self.row_data[:split_point]
        test_row_data = self.row_data[split_point:]

        for data_point in train_row_data:  # row_data = [ [x, edge_index, edge_attr], [x, edge_index, edge_attr], ... ]
            data_example = Data(
                x=torch.tensor(data_point[0], dtype=torch.float),
                edge_index=torch.tensor(data_point[1], dtype=torch.long).contiguous(),  
                edge_attr=torch.tensor(data_point[2], dtype=torch.float),
            )
            data_example.num_nodes = len(data_point[0])
            self.train_data_list.append(data_example)

        for data_point in test_row_data:  # row_data = [ [x, edge_index, edge_attr], [x, edge_index, edge_attr], ... ]
            data_example = Data(
                x=torch.tensor(data_point[0], dtype=torch.float),
                edge_index=torch.tensor(data_point[1], dtype=torch.long).contiguous(),  
                edge_attr=torch.tensor(data_point[2], dtype=torch.float),
            )
            data_example.num_nodes = len(data_point[0])
            self.test_data_list.append(data_example)
        train_data = DataLoader(self.train_data_list, batch_size=batch_size, shuffle=True)
        test_data = DataLoader(self.test_data_list, batch_size=batch_size, shuffle=True)
        return train_data, test_data


def edge_attr_to_node_attr(edge_index, edge_attr, num_nodes):
    node_attr = torch.zeros(num_nodes, edge_attr.size(1), device=edge_attr.device)
    edge_count = torch.zeros(num_nodes, device=edge_attr.device)

    for i in range(edge_index.size(1)):
        node_attr[edge_index[0, i]] += edge_attr[i]
        node_attr[edge_index[1, i]] += edge_attr[i]
        edge_count[edge_index[0, i]] += 1
        edge_count[edge_index[1, i]] += 1

    edge_count += 1e-8
    node_attr = node_attr / edge_count.view(-1, 1)

    return node_attr


# Why we also use GCN as decoder:
# Consistent. Otherwise, the Decoder may not be able to reconstruct the original graphics well
class MyDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyDecoder, self).__init__()

        self.conv1 = GCNConv(input_size, hidden_size)
        self.conv2 = GCNConv(hidden_size, output_size)

    def forward(self, z, edge_index):
        x = F.relu(self.conv1(z, edge_index))
        x = self.conv2(x, edge_index)
        return x


class EncoderDecoder3(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(EncoderDecoder3, self).__init__()

        # all size define here is variable.shape[0]
        self.conv1 = GCNConv(input_size, hidden_size)
        self.conv2 = GCNConv(hidden_size, output_size)

        self.decoder = MyDecoder(output_size, hidden_size, input_size)

    def encode(self, x, edge_index, edge_attr):
        num_nodes, num_features = x.shape
        edge_attr = edge_attr_to_node_attr(edge_index, edge_attr, num_nodes)
        x = torch.cat([x, edge_attr], dim=1)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        z = F.relu(self.conv2(x, edge_index))
        # Although edge_index has been used in conv1,
        # the value of edge_index needs to be updated
        # in each convolutional layer so that the next convolutional layer
        # can correctly identify the connection between nodes
        f = global_max_pool(z, batch=None)
        f = torch.squeeze(f)
        return [f, z]  # shape in (num_nodes, output_size)
        # In encode, edge_index is treated as PASSING PARAMETER rather than include in x as feature
        # And is just for message passing, but STILL will effect the output

    def decode(self, z, edge_index):
        x_attr = self.decoder(z, edge_index)
        return x_attr

    # TILL HERE, use functions encode and decode
    def forward(self, x, edge_index, edge_attr):
        z = self.encode(x, edge_index, edge_attr)
        x_attr = self.decode(z, edge_index)
        return x_attr
