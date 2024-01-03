import copy
import math
import random
import statistics

import gym
import numpy as np
import torch
from gym import spaces

from src.SFC.CODER.EDcode.GNN1 import EncoderDecoder1
from src.SFC.CODER.EDcode.GNN2 import EncoderDecoder2
from src.SFC.CODER.EDcode.GNN3 import EncoderDecoder3

from src.SFC.DRL.Network import Network, Vertex

from torch_geometric.data import Data

image = "/code/src/SFC/Figures/Net_Arch.png"

class SfcEnvironment(gym.Env[np.array, np.array]):  # Obs, action architecture
    def __init__(self):
        # network setup
        self.state = None
        self.action = 9
        self.output_size = 32
        self.network_index = 0
        self.node_number = (
            0  # it is used when append to self.requests to determine how many request will be generated.
        )
        self.network = Network()
        self.nodes = self.network.get_node_list()
        self.requests = []
        self.modified = []
        self.x = None
        self.edge_index = None
        self.edge_attr = None
        self.action_space = spaces.Box(low=0, high=1, shape=(self.action,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.output_size,), dtype=np.float32)
        self.traffic_decreasing_ratio = 0.3
        self.unified_switching_power = 30
        self.current_processing_requests = (
            0 
        )
        self.total_effect_request = 0 
        self.current_delay = 0 
        self.hop = 1  
        self.current_node = None 
        self.source_node = None 
        self.success_ratio = None 
        self.new_request = True
        self.destination = None
        self.computing_resource = 100
        self.bandwidth_resource = 35
        self.power_recorder_0 = [] 
        self.power_recorder_1 = []
        self.power_recorder_2 = []
        self.power_recorder_3 = []
        self.power_recorder_0_all = []
        self.power_recorder_1_all = []
        self.power_recorder_2_all = []
        self.power_recorder_3_all = []
        self.slicing_window = 6
        self.failure = 0  
        self.flag = 0
        self.individul_hop = 0  
        self.episode = [] 
        ############################################
        self.lev0 = -0.2 
        self.lev1 = 0  
        self.lev2 = 0.2  
        self.lev3 = 0.4  
        self.lev4_0 = -0.4  
        self.lev4_1 = 1  
        self.lev4_2 = 1  

    def build_network(self):
        self.network_index = random.randrange(0, 3)
        self.network_index = 0
        self.network = Network()
        self.current_processing_requests = 0
        self.total_effect_request = 0
        self.current_delay = 0
        self.hop = 1
        self.new_request = True
        self.failure = 0
        self.flag = 0
        self.individul_hop = 0
        self.success_ratio = None

        if self.network_index == 0:
            self.node_number = 4
            self.destination = "Node4"
            self.network.add_node("Node0", random.randrange(self.computing_resource - 30, self.computing_resource), 1)
            self.network.add_node("Node1", random.randrange(self.computing_resource - 30, self.computing_resource), 0)
            self.network.add_node("Node2", random.randrange(self.computing_resource - 30, self.computing_resource), 1)
            self.network.add_node("Node3", random.randrange(self.computing_resource - 30, self.computing_resource), 0)
            self.network.add_node("Node4", 10000, 1)

            self.network.add_edge(
                "Node0", "Node1", random.randrange(self.bandwidth_resource - 5, self.bandwidth_resource), 11
            )
            self.network.add_edge(
                "Node0", "Node3", random.randrange(self.bandwidth_resource - 5, self.bandwidth_resource), 10
            )
            self.network.add_edge(
                "Node1", "Node2", random.randrange(self.bandwidth_resource - 5, self.bandwidth_resource), 12
            )
            self.network.add_edge(
                "Node2", "Node3", random.randrange(self.bandwidth_resource - 5, self.bandwidth_resource), 7
            )
            self.network.add_edge("Node1", "Node4", self.bandwidth_resource * 3, 20)
            self.network.add_edge("Node2", "Node4", self.bandwidth_resource * 3, 20)

        elif self.network_index == 1:
            self.node_number = 5
            self.destination = "Node5"
            self.network.add_node("Node0", random.randrange(self.computing_resource - 30, self.computing_resource), 1)
            self.network.add_node("Node1", random.randrange(self.computing_resource - 30, self.computing_resource), 0)
            self.network.add_node("Node2", random.randrange(self.computing_resource - 30, self.computing_resource), 1)
            self.network.add_node("Node3", random.randrange(self.computing_resource - 30, self.computing_resource), 1)
            self.network.add_node("Node4", random.randrange(self.computing_resource - 30, self.computing_resource), 0)
            self.network.add_node("Node5", 10000, 1)

            self.network.add_edge(
                "Node0", "Node1", random.randrange(self.bandwidth_resource - 5, self.bandwidth_resource), 10
            )
            self.network.add_edge(
                "Node0", "Node2", random.randrange(self.bandwidth_resource - 5, self.bandwidth_resource), 12
            )
            self.network.add_edge(
                "Node0", "Node4", random.randrange(self.bandwidth_resource - 5, self.bandwidth_resource), 11
            )
            self.network.add_edge(
                "Node1", "Node2", random.randrange(self.bandwidth_resource - 5, self.bandwidth_resource), 7
            )
            self.network.add_edge(
                "Node2", "Node3", random.randrange(self.bandwidth_resource - 5, self.bandwidth_resource), 8
            )
            self.network.add_edge(
                "Node3", "Node4", random.randrange(self.bandwidth_resource - 5, self.bandwidth_resource), 7
            )
            self.network.add_edge("Node3", "Node5", self.bandwidth_resource * 3, 20)

        elif self.network_index == 2:
            self.node_number = 5
            self.destination = "Node5"
            self.network.add_node("Node0", random.randrange(self.computing_resource - 30, self.computing_resource), 1)
            self.network.add_node("Node1", random.randrange(self.computing_resource - 30, self.computing_resource), 1)
            self.network.add_node("Node2", random.randrange(self.computing_resource - 30, self.computing_resource), 0)
            self.network.add_node("Node3", random.randrange(self.computing_resource - 30, self.computing_resource), 1)
            self.network.add_node("Node4", random.randrange(self.computing_resource - 30, self.computing_resource), 0)
            self.network.add_node("Node5", 10000, 1)

            self.network.add_edge(
                "Node0", "Node1", random.randrange(self.bandwidth_resource - 5, self.bandwidth_resource), 13
            )
            self.network.add_edge(
                "Node0", "Node4", random.randrange(self.bandwidth_resource - 5, self.bandwidth_resource), 8
            )
            self.network.add_edge(
                "Node1", "Node2", random.randrange(self.bandwidth_resource - 5, self.bandwidth_resource), 9
            )
            self.network.add_edge(
                "Node1", "Node3", random.randrange(self.bandwidth_resource - 5, self.bandwidth_resource), 9
            )
            self.network.add_edge(
                "Node1", "Node4", random.randrange(self.bandwidth_resource - 5, self.bandwidth_resource), 7
            )
            self.network.add_edge(
                "Node2", "Node3", random.randrange(self.bandwidth_resource - 5, self.bandwidth_resource), 11
            )
            self.network.add_edge(
                "Node3", "Node4", random.randrange(self.bandwidth_resource - 5, self.bandwidth_resource), 10
            )
            self.network.add_edge("Node3", "Node5", self.bandwidth_resource * 3, 20)

        ########### used for incremeanl learning ############
        elif self.network_index == 3:
            self.node_number = 5
            self.destination = "Node5"
            self.network.add_node("Node0", random.randrange(self.computing_resource - 30, self.computing_resource), 1)
            self.network.add_node("Node1", random.randrange(self.computing_resource - 30, self.computing_resource), 0)
            self.network.add_node("Node2", random.randrange(self.computing_resource - 30, self.computing_resource), 1)
            self.network.add_node("Node3", random.randrange(self.computing_resource - 30, self.computing_resource), 0)
            self.network.add_node("Node4", random.randrange(self.computing_resource - 30, self.computing_resource), 1)
            self.network.add_node("Node5", 10000, 1)

            self.network.add_edge(
                "Node0", "Node1", random.randrange(self.bandwidth_resource - 5, self.bandwidth_resource), 11
            )
            self.network.add_edge(
                "Node0", "Node3", random.randrange(self.bandwidth_resource - 5, self.bandwidth_resource), 10
            )
            self.network.add_edge(
                "Node0", "Node4", random.randrange(self.bandwidth_resource - 5, self.bandwidth_resource), 8
            )
            self.network.add_edge(
                "Node1", "Node2", random.randrange(self.bandwidth_resource - 5, self.bandwidth_resource), 12
            )
            self.network.add_edge(
                "Node2", "Node3", random.randrange(self.bandwidth_resource - 5, self.bandwidth_resource), 7
            )
            self.network.add_edge(
                "Node3", "Node4", random.randrange(self.bandwidth_resource - 5, self.bandwidth_resource), 9
            )
            self.network.add_edge("Node1", "Node5", self.bandwidth_resource * 3, 20)
            self.network.add_edge("Node2", "Node5", self.bandwidth_resource * 3, 20)

        self.nodes = self.network.get_node_list()


    ###################### for split option 7.2 O-RAN #######################
    def generate_requests_information(self):
        prob = random.random()  # Generate a random number between 0 and 1
        if prob < 1 / 4:  # uRLLC
            # If the number is less than 0.5, generate a random number between 10 and 20
            return (
                0,  # service_type
                random.randrange(13, 20),  # fronthal_delay
                random.randint(30, 40),  # end-to-end delay
                random.randint(15, 25),  # DU
                random.randint(5, 10),  # CU-UP
                random.randint(5, 10),  # CU-CP
                random.randint(15, 25),  # UPF
                random.randint(2, 4),  # bandwidth
            )
        elif 1 / 4 <= prob < 2 / 4:  # eMBB
            return (
                1,  # service_type
                random.randrange(20, 25),  # fronthal_delay
                random.randint(30, 40),  # midhaul delay
                random.randint(15, 25),  # DU
                random.randint(15, 25),  # CU-UP
                random.randint(5, 10),  # CU-CP
                0,  # UPF
                random.randint(4, 7),  # bandwidth
            )
        elif 2 / 4 <= prob < 3 / 4:  # mMTC
            return (
                2,  # service_type
                random.randrange(20, 25),  # fronthal_delay
                random.randint(30, 40),  # midhaul delay
                random.randint(15, 25),  # DU
                random.randint(5, 10),  # CU-UP
                random.randint(15, 25),  # CU-CP
                0,  # UPF
                random.randint(4, 7),  # bandwidth
            )
        elif 3 / 4 < prob:  # no requests
            return (
                3,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            )

    def generate_requests(self):
        self.requests = []
        for _ in range(self.node_number): 
            (
                service_type,
                fronthal_delay,
                end_to_end_delay,
                DU_computing,
                CUUP_computing,
                CUCP_computing,
                UPF_computing,
                data_size,
            ) = self.generate_requests_information()
            self.requests.append(
                [
                    service_type,  # uRLLC/eMBB/mMTC
                    fronthal_delay,  # fronthaul delay
                    end_to_end_delay,  # E2E delay for urllc, midhaul delay for embb and mmtc (till CU-CP)
                    DU_computing,  # DU
                    CUUP_computing,  # CU-UP
                    CUCP_computing,  # CU-CP
                    UPF_computing,  # UPF
                    data_size,  # bandwidth
                ]
            )
            if service_type != 3:
                self.total_effect_request += 1

       
        def find_last_one_index(lst):
            for i in range(len(lst) - 1, -1, -1):
                if lst[i] != 3:
                    return i

        def are_all_elements_three(lst):
            return all(element == 3 for element in lst)

        types_of_request = []
        for req in self.requests: 
            types_of_request.append(req[0])

        if are_all_elements_three(types_of_request):
            self.reset()
        else:
            last_effect_request = find_last_one_index(types_of_request)

            while self.current_processing_requests <= last_effect_request:
                if self.requests[self.current_processing_requests][0] != 3:  
                    break
                else:
                    self.current_processing_requests += 1
            self.source_node = "Node" + str(self.current_processing_requests)
            self.current_node = self.source_node

    def ReadNet(self):
        # netwrok information to encoder input shape
        key_list = list(self.network.get_node_list())
        # print(key_list)
        self.modified = []  # used to create self.x

        for node_idx in range(len(key_list[:-1])):
            normarlized_request = copy.deepcopy(self.requests[node_idx])
            normarlized_request[0] = normarlized_request[0] / 3
            normarlized_request[1] = normarlized_request[1] / 25  ############################### for split 7.2 ###################################
            normarlized_request[2] = normarlized_request[2] / 40
            normarlized_request[3] = normarlized_request[3] / 25
            normarlized_request[4] = normarlized_request[4] / 25
            normarlized_request[5] = normarlized_request[5] / 25
            normarlized_request[6] = normarlized_request[6] / 25
            normarlized_request[7] = normarlized_request[7] / 7 ############################### for split 7.2 ###################################

            normarlized_request.append(self.network.node_list[key_list[node_idx]].node_type)  
            normarlized_request.append(self.current_processing_requests / self.node_number)  
            normarlized_request.append(
                self.network.node_list[key_list[node_idx]].get_resource() / self.computing_resource
            )  

            if key_list[node_idx] == self.current_node:  
                normarlized_request.append(1)
            else:
                normarlized_request.append(0)

            self.modified.append(normarlized_request)  

        if self.current_node == key_list[-1]: 
            self.modified.append(
                [1, 0, 0, 0, 0, 0, 0, 0, 1, self.current_processing_requests / self.node_number, 2, 1]
            )  
        else:
            self.modified.append(
                [1, 0, 0, 0, 0, 0, 0, 0, 1, self.current_processing_requests / self.node_number, 2, 0]
            ) 

        # extract the network information and put it into self.x, edge_index and edge_attr
        self.x = torch.tensor(self.modified, dtype=torch.float)
        if self.network_index == 0:
            self.edge_index = torch.tensor([[0, 1, 2, 3, 1, 2], [1, 2, 3, 0, 4, 4]], dtype=torch.long).contiguous()
            edge_attr = []
            for edge in [[0, 1], [1, 2], [2, 3], [3, 0], [1, 4], [2, 4]]:
                edge_BW = (
                    self.network.node_list["Node" + str(edge[0])].get_bandwidth("Node" + str(edge[1]))
                    / self.bandwidth_resource
                )
                edge_DT = self.network.node_list["Node" + str(edge[0])].get_distance("Node" + str(edge[1]))
                edge_attr.append([edge_DT, edge_BW])
            self.edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        elif self.network_index == 1:
            self.edge_index = torch.tensor(
                [[0, 1, 2, 3, 4, 0, 3], [1, 2, 3, 4, 0, 2, 5]], dtype=torch.long
            ).contiguous()
            edge_attr = []
            for edge in [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0], [0, 2], [3, 5]]:
                edge_BW = (
                    self.network.node_list["Node" + str(edge[0])].get_bandwidth("Node" + str(edge[1]))
                    / self.bandwidth_resource
                )
                edge_DT = self.network.node_list["Node" + str(edge[0])].get_distance("Node" + str(edge[1]))
                edge_attr.append([edge_DT, edge_BW])
            self.edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        elif self.network_index == 2:
            self.edge_index = torch.tensor(
                [[0, 1, 2, 3, 4, 4, 1, 3], [1, 2, 3, 4, 0, 1, 3, 5]], dtype=torch.long
            ).contiguous()
            edge_attr = []
            for edge in [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0], [4, 1], [1, 3], [3, 5]]:
                edge_BW = (
                    self.network.node_list["Node" + str(edge[0])].get_bandwidth("Node" + str(edge[1]))
                    / self.bandwidth_resource
                )
                edge_DT = self.network.node_list["Node" + str(edge[0])].get_distance("Node" + str(edge[1]))
                edge_attr.append([edge_DT, edge_BW])
            self.edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        elif self.network_index == 3:
            self.edge_index = torch.tensor(
                [[0, 1, 2, 3, 0, 0, 1, 2], [1, 2, 3, 4, 3, 4, 5, 5]], dtype=torch.long
            ).contiguous()
            edge_attr = []
            for edge in [[0, 1], [1, 2], [2, 3], [3, 4], [0, 3], [0, 4], [1, 5], [2, 5]]:
                edge_BW = (
                    self.network.node_list["Node" + str(edge[0])].get_bandwidth("Node" + str(edge[1]))
                    / self.bandwidth_resource
                )
                edge_DT = self.network.node_list["Node" + str(edge[0])].get_distance("Node" + str(edge[1]))
                edge_attr.append([edge_DT, edge_BW])
            self.edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        network_data = Data(
            x=torch.tensor(self.x, dtype=torch.float),
            edge_index=torch.tensor(self.edge_index, dtype=torch.long).contiguous(),
            edge_attr=torch.tensor(self.edge_attr, dtype=torch.float),
        )
        network_data.num_nodes = len(self.x[0])

        hidden_size = 32
        output_size = self.output_size
        # then to features
        if self.network_index == 0:
            input_size = self.x.shape[1] + self.edge_attr.shape[1]
            model = EncoderDecoder1(input_size, hidden_size, output_size)
            model.load_state_dict(
                torch.load(
                    "/code/src/SFC/CODER/EDcode/Network_" ############################### for split 7.2 ###################################
                    + str(self.network_index + 1)
                    + "/model_"
                    + str(self.network_index + 1)
                    + ".pth"
                )
            )
        elif self.network_index == 1:
            input_size = self.x.shape[1] + self.edge_attr.shape[1]
            model = EncoderDecoder2(input_size, hidden_size, output_size)
            model.load_state_dict(
                torch.load(
                    "/code/src/SFC/CODER/EDcode/Network_" ############################### for split 7.2 ###################################
                    + str(self.network_index + 1)
                    + "/model_"
                    + str(self.network_index + 1)
                    + ".pth"
                )
            )
        elif self.network_index == 2:
            input_size = self.x.shape[1] + self.edge_attr.shape[1]
            model = EncoderDecoder3(input_size, hidden_size, output_size)
            model.load_state_dict(
                torch.load(
                    "/code/src/SFC/CODER/EDcode/Network_" ############################### for split 7.2 ###################################
                    + str(self.network_index + 1)
                    + "/model_"
                    + str(self.network_index + 1)
                    + ".pth"
                )
            )

        [self.state, z] = model.encode(network_data.x, network_data.edge_index, network_data.edge_attr)

    def step(self, action):
        reward, terminated = self.action_to_deployment(action)
        ### debug ###
        # reward = 1
        # deployment = 1
        self.ReadNet()
        return self.state.detach().numpy().astype(np.float32), reward, terminated, {}

    # the Out degree in our set up is 4， which means there will be 4 directions of the action
    # another action is to decide put function

    def action_to_deployment(
        self, action
    ): 

        def find_last_one_index(lst): 
            for i in range(len(lst) - 1, -1, -1):
                if lst[i] != 3:
                    return i

        def calculate_power(Node):
            load = 1 - self.network.node_list[Node].get_resource() / self.computing_resource
            power = 130 * load**2 + 90
            return power

        def store_data(data_list, new_data, number):
            data_list.append(new_data)

            if len(data_list) > number:
                del data_list[0]

        def bottom_avg(data_list):
            new_data_list = copy.deepcopy(data_list)
            new_data_list.sort()
            # n = len(new_data_list)
            # percent = int(n * 0.1)
            # avg = sum(new_data_list[:percent]) / percent
            minimun = new_data_list[0]
            return minimun

        types_of_request = []
        for req in self.requests:  
            types_of_request.append(req[0])
        last_effect_request = find_last_one_index(types_of_request)
        # # print(f"last_effect_request: {last_effect_request}")

        if self.new_request == True: 
            # print('------------------self.new_request == True-------------------')
            self.new_request = False
            while self.current_processing_requests <= last_effect_request:
                if self.requests[self.current_processing_requests][0] != 3:  
                    break
                else:
                    self.current_processing_requests += 1
            self.source_node = "Node" + str(self.current_processing_requests)
            # # print(
            #     f"current_processing_requests: {self.current_processing_requests}"
            # )  
            self.current_node = self.source_node
            self.current_delay = 0

            neighbor_index = action // 2
            if neighbor_index != 4:  
                # print('----------neighbor_index != 4:-----------')
                if neighbor_index < len( 
                    self.network.node_list[self.current_node].get_neighbors()
                ):  
                    selected_neighbor = list(self.network.node_list[self.current_node].get_neighbors())[neighbor_index]
                    # print('---------- valid action -----------')
                    deployment_decision = action % 2 
                    if deployment_decision == 1:
                        # print('--------deploy-------')
                        if (
                            self.current_delay
                            + self.network.node_list[self.current_node].get_distance(selected_neighbor)
                            <= self.requests[self.current_processing_requests][1]
                        ):  
                            # print('------fronthaul satisfiesd-------')
                            if (
                                self.network.node_list[selected_neighbor].get_resource()
                                >= self.requests[self.current_processing_requests][3]
                                and self.network.node_list[selected_neighbor].node_type == 1
                            ):  
                                # print('------enough computing-------')
                                if (
                                    self.network.node_list[self.current_node].get_bandwidth(selected_neighbor)
                                    >= self.requests[self.current_processing_requests][7]
                                ):  
                                    # print('------enough bandwidth-------')
                                    self.current_delay += self.network.node_list[self.current_node].get_distance(
                                        selected_neighbor
                                    )  
                                    self.hop += 1  
                                    self.network.node_list[selected_neighbor].change_computing(
                                        -self.requests[self.current_processing_requests][3]
                                    ) 
                                    self.network.node_list[self.current_node].change_bandwidth(
                                        selected_neighbor, -self.requests[self.current_processing_requests][7]
                                    )  
                                    self.requests[self.current_processing_requests][7] = (
                                        self.requests[self.current_processing_requests][7]
                                        * self.traffic_decreasing_ratio
                                    )  
                                    self.flag += 1
                                    self.current_node = selected_neighbor  
                                    reward = self.lev2
                                    terminate = False
                                    return reward, terminate
                                else: 
                                    # print('------lack computing-------')
                                    if self.current_processing_requests == last_effect_request:
                                        # print('------last request-------')
                                        self.failure += 1
                                        self.success_ratio = 1 - self.failure / self.total_effect_request
                                        if self.success_ratio <= 1 / 2:
                                            reward = self.lev4_0
                                        elif 1 / 2 < self.success_ratio:
                                            reward = self.lev4_1 * self.success_ratio
                                        terminate = True
                                        self.episode.append(reward)
                                        return reward, terminate
                                    else:
                                        # print('------not last request-------')
                                        reward = self.lev0
                                        self.flag = 0
                                        self.new_request = True
                                        self.current_processing_requests += 1
                                        terminate = False
                                        self.failure += 1
                                        return reward, terminate
                            else:
                                # print('------lack computing-------')
                                if self.current_processing_requests == last_effect_request:
                                    # print('------last request-------')
                                    self.failure += 1
                                    self.success_ratio = 1 - self.failure / self.total_effect_request
                                    if self.success_ratio <= 1 / 2:
                                        reward = self.lev4_0
                                    elif 1 / 2 < self.success_ratio:
                                        reward = self.lev4_1 * self.success_ratio
                                    terminate = True
                                    self.episode.append(reward)
                                    return reward, terminate
                                else:
                                    # print('------not last request-------')
                                    reward = self.lev0
                                    self.flag = 0
                                    self.new_request = True
                                    self.current_processing_requests += 1
                                    terminate = False
                                    self.failure += 1
                                    return reward, terminate
                        else:
                            # print('------fronthaul failed-------')
                            if self.current_processing_requests == last_effect_request:
                                # print('------last request-------')
                                self.failure += 1
                                self.success_ratio = 1 - self.failure / self.total_effect_request
                                if self.success_ratio <= 1 / 2:
                                    reward = self.lev4_0
                                elif 1 / 2 < self.success_ratio:
                                    reward = self.lev4_1 * self.success_ratio
                                terminate = True
                                self.episode.append(reward)
                                return reward, terminate
                            else:
                                # print('------not last request-------')
                                reward = self.lev0
                                self.flag = 0
                                self.new_request = True
                                self.current_processing_requests += 1
                                terminate = False
                                self.failure += 1
                                return reward, terminate
                    else: 
                        # print('--------no deploy-------')
                        if (
                            self.current_delay
                            + self.network.node_list[self.current_node].get_distance(selected_neighbor)
                            <= self.requests[self.current_processing_requests][1]
                        ):  
                            # print('------fronthaul satisfiesd-------')
                            if (
                                self.network.node_list[self.current_node].get_bandwidth(selected_neighbor)
                                >= self.requests[self.current_processing_requests][7]
                            ):  
                                # print('------bandwidth enough-------')
                                self.current_delay += self.network.node_list[self.current_node].get_distance(
                                    selected_neighbor
                                )  
                                self.hop += 1  
                                self.network.node_list[self.current_node].change_bandwidth(
                                    selected_neighbor, -self.requests[self.current_processing_requests][7]
                                )  
                                self.current_node = selected_neighbor
                                reward = self.lev1
                                terminate = False
                                return reward, terminate
                            else:
                                # print('------lack bandwidth-------')
                                if self.current_processing_requests == last_effect_request:
                                    # print('------last request-------')
                                    self.failure += 1
                                    self.success_ratio = 1 - self.failure / self.total_effect_request
                                    if self.success_ratio <= 1 / 2:
                                        reward = self.lev4_0
                                    elif 1 / 2 < self.success_ratio:
                                        reward = self.lev4_1 * self.success_ratio
                                    terminate = True
                                    self.episode.append(reward)
                                    return reward, terminate
                                else:
                                    # print('------not last request-------')
                                    reward = self.lev0
                                    self.flag = 0
                                    self.new_request = True
                                    self.current_processing_requests += 1
                                    terminate = False
                                    self.failure += 1
                                    return reward, terminate
                        else:
                            # print('------fronthaul failed-------')
                            if self.current_processing_requests == last_effect_request:
                                # print('------last request-------')
                                self.failure += 1
                                self.success_ratio = 1 - self.failure / self.total_effect_request
                                if self.success_ratio <= 1 / 2:
                                    reward = self.lev4_0
                                elif 1 / 2 < self.success_ratio:
                                    reward = self.lev4_1 * self.success_ratio
                                terminate = True
                                self.episode.append(reward)
                                return reward, terminate
                            else:
                                # print('------not last request-------')
                                reward = self.lev0
                                self.flag = 0
                                self.current_processing_requests += 1
                                terminate = False
                                self.failure += 1
                                return reward, terminate
                else:
                    # print('---------- UN valid action -----------')
                    if self.current_processing_requests == last_effect_request:
                        # print('------last request-------')
                        self.failure += 1
                        self.success_ratio = 1 - self.failure / self.total_effect_request
                        if self.success_ratio <= 1 / 2:
                            reward = self.lev4_0
                        elif 1 / 2 < self.success_ratio:
                            reward = self.lev4_1 * self.success_ratio
                        terminate = True
                        self.episode.append(reward)
                        return reward, terminate
                    else:
                        # print('------not last request-------')
                        reward = self.lev0
                        self.flag = 0
                        self.new_request = True
                        self.current_processing_requests += 1
                        terminate = False
                        self.failure += 1
                        return reward, terminate
            else:  
                # print('----------neighbor_index == 4:-----------')
                if (
                    self.network.node_list[self.current_node].get_resource()
                    >= self.requests[self.current_processing_requests][3]
                    and self.network.node_list[self.current_node].node_type == 1
                ):  
                    # print('------enough computing-------')
                    self.network.node_list[self.current_node].change_computing(
                        -self.requests[self.current_processing_requests][3]
                    ) 
                    self.requests[self.current_processing_requests][7] = (
                        self.requests[self.current_processing_requests][7] * self.traffic_decreasing_ratio
                    )  
                    reward = self.lev2
                    terminate = False
                    return reward, terminate
                else:
                    # print('------lack computing-------')
                    if self.current_processing_requests == last_effect_request:
                        self.failure += 1
                        self.success_ratio = 1 - self.failure / self.total_effect_request
                        if self.success_ratio <= 1 / 2:
                            reward = self.lev4_0
                        elif 1 / 2 < self.success_ratio:
                            reward = self.lev4_1 * self.success_ratio
                        terminate = True
                        self.episode.append(reward)
                        return reward, terminate
                    else:
                        reward = self.lev0
                        self.flag = 0
                        self.new_request = True
                        self.current_processing_requests += 1
                        terminate = False
                        self.failure += 1
                        return reward, terminate

        else:
            # print('------------------self.new_request == False-------------------')
            if self.current_processing_requests != last_effect_request:  
                # print('-------------not the last_effect_request-------------')
                terminate = False
                neighbor_index = action // 2
                if (
                    self.requests[self.current_processing_requests][0] == 0
                ):  # uRLLC
                    # print('--------- URLLC ---------')
                    if neighbor_index != 4: 
                        # print('----------neighbor_index != 4:-----------')
                        if neighbor_index < len(
                            self.network.node_list[self.current_node].get_neighbors()
                        ): 
                            # print('----------valid aciton-----------')
                            selected_neighbor = list(self.network.node_list[self.current_node].get_neighbors())[
                                neighbor_index
                            ]
                            deployment_decision = action % 2  

                           
                            if deployment_decision == 1:  
                                # print('----------deploy-----------')
                                if self.flag == 0:  
                                    # print('----------no DU yet-----------')
                                    if (
                                        self.current_delay
                                        + self.network.node_list[self.current_node].get_distance(selected_neighbor)
                                        <= self.requests[self.current_processing_requests][1]
                                    ):  
                                        # print('----------fronthaul satisfied-----------')
                                        if (
                                            self.network.node_list[selected_neighbor].get_resource()
                                            >= self.requests[self.current_processing_requests][self.flag + 3]
                                            and self.network.node_list[selected_neighbor].node_type == 1
                                        ):  
                                            # print('----------enough computing-----------')
                                            if (
                                                self.network.node_list[self.current_node].get_bandwidth(
                                                    selected_neighbor
                                                )
                                                >= self.requests[self.current_processing_requests][7]
                                            ):  
                                                # print('----------enough bandwidth-----------')
                                                self.current_delay += self.network.node_list[
                                                    self.current_node
                                                ].get_distance(
                                                    selected_neighbor
                                                )  
                                                self.hop += 1  
                                                self.network.node_list[selected_neighbor].change_computing(
                                                    -self.requests[self.current_processing_requests][self.flag + 3]
                                                ) 
                                                self.network.node_list[self.current_node].change_bandwidth(
                                                    selected_neighbor,
                                                    -self.requests[self.current_processing_requests][7],
                                                )  
                                                self.requests[self.current_processing_requests][7] = (
                                                    self.requests[self.current_processing_requests][7]
                                                    * self.traffic_decreasing_ratio
                                                )  
                                                self.current_node = selected_neighbor
                                                self.flag += 1
                                                reward = self.lev2
                                                return reward, terminate
                                            else:
                                                # print('----------lack bandwidth-----------')
                                                reward = self.lev0
                                                self.flag = 0
                                                self.new_request = True
                                                self.current_processing_requests += 1
                                                self.failure += 1
                                                return reward, terminate
                                        else:
                                            # print('----------lack comupting-----------')
                                            reward = self.lev0
                                            self.flag = 0
                                            self.new_request = True
                                            self.current_processing_requests += 1
                                            self.failure += 1
                                            return reward, terminate
                                    else:
                                        # print('----------fronthaul fail-----------')
                                        reward = self.lev0
                                        self.flag = 0
                                        self.new_request = True
                                        self.current_processing_requests += 1
                                        self.failure += 1
                                        return reward, terminate
                                else:  
                                    # print('----------have DU, do cu or upf now-----------')
                                    if (
                                        self.current_delay
                                        + self.network.node_list[self.current_node].get_distance(selected_neighbor)
                                        <= self.requests[self.current_processing_requests][2]
                                    ):  
                                        # print('----------end-to-end satisfied-----------')
                                        if (
                                            self.network.node_list[selected_neighbor].get_resource()
                                            >= self.requests[self.current_processing_requests][self.flag + 3]
                                        ):  
                                            # print('----------enough computing-----------')
                                            if (
                                                self.network.node_list[self.current_node].get_bandwidth(
                                                    selected_neighbor
                                                )
                                                >= self.requests[self.current_processing_requests][7]
                                            ): 
                                                # print('----------enough bandwidth-----------')
                                                self.current_delay += self.network.node_list[
                                                    self.current_node
                                                ].get_distance(
                                                    selected_neighbor
                                                )  
                                                self.hop += 1  
                                                self.network.node_list[selected_neighbor].change_computing(
                                                    -self.requests[self.current_processing_requests][self.flag + 3]
                                                )  
                                                self.network.node_list[self.current_node].change_bandwidth(
                                                    selected_neighbor,
                                                    -self.requests[self.current_processing_requests][7],
                                                )  
                                                self.requests[self.current_processing_requests][7] = (
                                                    self.requests[self.current_processing_requests][7]
                                                    * self.traffic_decreasing_ratio
                                                )  
                                                self.flag += 1
                                                self.current_node = selected_neighbor
                                                if self.flag == 4:
                                                    # print('----------UPF place-----------')
                                                    reward = self.lev3
                                                    self.flag = 0
                                                    self.new_request = True
                                                    self.current_processing_requests += 1
                                                    return reward, terminate
                                                else:
                                                    # print('----------not UPF-----------')
                                                    reward = self.lev2
                                                    return reward, terminate
                                            else:
                                                # print('----------lack bandwidth-----------')
                                                reward = self.lev0
                                                self.flag = 0
                                                self.new_request = True
                                                self.current_processing_requests += 1
                                                self.failure += 1
                                                return reward, terminate
                                        else:
                                            # print('----------lack computing-----------')
                                            reward = self.lev0
                                            self.flag = 0
                                            self.new_request = True
                                            self.current_processing_requests += 1
                                            self.failure += 1
                                            return reward, terminate
                                    else:
                                        # print('----------end-to-end fail-----------')
                                        reward = self.lev0
                                        self.flag = 0
                                        self.new_request = True
                                        self.current_processing_requests += 1
                                        self.failure += 1
                                        return reward, terminate
                            else:  
                                # print('----------no deploy-----------')
                                if self.flag == 0:  
                                    # print('----------no DU yet-----------')
                                    if (
                                        self.current_delay
                                        + self.network.node_list[self.current_node].get_distance(selected_neighbor)
                                        <= self.requests[self.current_processing_requests][1]
                                    ): 
                                        # print('----------fronthaul satisfied-----------')
                                        if (
                                            self.network.node_list[self.current_node].get_bandwidth(selected_neighbor)
                                            >= self.requests[self.current_processing_requests][7]
                                        ): 
                                            # print('----------bandwidth satisfied-----------')
                                            self.current_delay += self.network.node_list[
                                                self.current_node
                                            ].get_distance(
                                                selected_neighbor
                                            )  
                                            self.hop += 1  
                                            self.network.node_list[self.current_node].change_bandwidth(
                                                selected_neighbor, -self.requests[self.current_processing_requests][7]
                                            )  
                                            reward = self.lev1
                                            self.current_node = selected_neighbor
                                            return reward, terminate
                                        else:
                                            # print('----------lack bandwidth-----------')
                                            reward = self.lev0
                                            self.flag = 0
                                            self.new_request = True
                                            self.current_processing_requests += 1
                                            self.failure += 1
                                            return reward, terminate
                                    else:
                                        # print('----------fronthaul fail-----------')
                                        reward = self.lev0
                                        self.flag = 0
                                        self.new_request = True
                                        self.current_processing_requests += 1
                                        self.failure += 1
                                        return reward, terminate
                                else:  
                                    # print('----------DU done, cu upf now-----------')
                                    if (
                                        self.current_delay
                                        + self.network.node_list[self.current_node].get_distance(selected_neighbor)
                                        <= self.requests[self.current_processing_requests][2]
                                    ):  
                                        # print('----------end-to-end satisfied-----------')
                                        if (
                                            self.network.node_list[self.current_node].get_bandwidth(selected_neighbor)
                                            >= self.requests[self.current_processing_requests][7]
                                        ):  
                                            # print('----------enough bandwidth-----------')
                                            self.current_delay += self.network.node_list[
                                                self.current_node
                                            ].get_distance(
                                                selected_neighbor
                                            )  
                                            self.hop += 1  
                                            self.network.node_list[self.current_node].change_bandwidth(
                                                selected_neighbor, -self.requests[self.current_processing_requests][7]
                                            )  
                                            reward = self.lev1
                                            self.current_node = selected_neighbor
                                            return reward, terminate
                                        else:
                                            # print('----------lack bandwidth-----------')
                                            reward = self.lev0
                                            self.flag = 0
                                            self.new_request = True
                                            self.current_processing_requests += 1
                                            self.failure += 1
                                            return reward, terminate
                                    else:
                                        # print('----------end-to-end fail-----------')
                                        reward = self.lev0
                                        self.flag = 0
                                        self.new_request = True
                                        self.current_processing_requests += 1
                                        self.failure += 1
                                        return reward, terminate
                        else:  
                            # print('----------UN valid action-----------')
                            reward = self.lev0
                            self.flag = 0
                            self.new_request = True
                            self.current_processing_requests += 1
                            self.failure += 1
                            return reward, terminate
                    else: 
                        # print('----------neighbor_index == 4:-----------')
                        if self.flag == 0:  
                            # print('----------no DU yet-----------')
                            if (
                                self.network.node_list[self.current_node].get_resource()
                                >= self.requests[self.current_processing_requests][self.flag + 3]
                                and self.network.node_list[self.current_node].node_type == 1
                            ):  
                                # print('----------enough computing-----------')
                                self.network.node_list[self.current_node].change_computing(
                                    -self.requests[self.current_processing_requests][self.flag + 3]
                                ) 
                                self.requests[self.current_processing_requests][7] = (
                                    self.requests[self.current_processing_requests][7] * self.traffic_decreasing_ratio
                                )  
                                self.flag += 1
                                reward = self.lev2
                                return reward, terminate
                            else:
                                # print('----------lack computing-----------')
                                reward = self.lev0
                                self.flag = 0
                                self.new_request = True
                                self.current_processing_requests += 1
                                self.failure += 1
                                return reward, terminate
                        else:  
                            # print('----------DU done, cu, upf now-----------')
                            if (
                                self.network.node_list[self.current_node].get_resource()
                                >= self.requests[self.current_processing_requests][self.flag + 3]
                            ): 
                                # print('----------enough computing-----------')
                                self.network.node_list[self.current_node].change_computing(
                                    -self.requests[self.current_processing_requests][self.flag + 3]
                                )  
                                self.requests[self.current_processing_requests][7] = (
                                    self.requests[self.current_processing_requests][7] * self.traffic_decreasing_ratio
                                ) 
                                self.flag += 1
                                if self.flag == 4:
                                    self.flag = 0
                                    self.new_request = True
                                    self.current_processing_requests += 1
                                    reward = self.lev3
                                    return reward, terminate
                                else:
                                    reward = self.lev2
                                    return reward, terminate
                            else:
                                # print('----------lack computing-----------')
                                reward = self.lev0
                                self.flag = 0
                                self.new_request = True
                                self.current_processing_requests += 1
                                self.failure += 1
                                return reward, terminate
                else: 
                    # print('---------------- eMBB & mMTC ----------------')
                    if neighbor_index != 4: 
                        # print('----------neighbor_index != 4:-----------')
                        if neighbor_index < len(
                            self.network.node_list[self.current_node].get_neighbors()
                        ): 
                            # print('----------valid actoin-----------')
                            selected_neighbor = list(self.network.node_list[self.current_node].get_neighbors())[
                                neighbor_index
                            ]
                            deployment_decision = action % 2  
                           
                            if deployment_decision == 1: 
                                # print('----------deploy-----------')
                                if self.flag == 0: 
                                    # print('----------no DU yet-----------')
                                    if (
                                        self.current_delay
                                        + self.network.node_list[self.current_node].get_distance(selected_neighbor)
                                        <= self.requests[self.current_processing_requests][1]
                                    ):  
                                        # print('----------fronthaul satisfied-----------')
                                        if (
                                            self.network.node_list[selected_neighbor].get_resource()
                                            >= self.requests[self.current_processing_requests][self.flag + 3]
                                            and self.network.node_list[selected_neighbor].node_type == 1
                                        ):  
                                            # print('----------computing satisfied-----------')
                                            if (
                                                self.network.node_list[self.current_node].get_bandwidth(
                                                    selected_neighbor
                                                )
                                                >= self.requests[self.current_processing_requests][7]
                                            ):
                                                # print('----------bandwidth satisfied-----------')
                                                self.current_delay += self.network.node_list[
                                                    self.current_node
                                                ].get_distance(
                                                    selected_neighbor
                                                )  
                                                self.hop += 1  
                                                self.network.node_list[selected_neighbor].change_computing(
                                                    -self.requests[self.current_processing_requests][self.flag + 3]
                                                )  
                                                self.network.node_list[self.current_node].change_bandwidth(
                                                    selected_neighbor,
                                                    -self.requests[self.current_processing_requests][7],
                                                )  
                                                self.requests[self.current_processing_requests][7] = (
                                                    self.requests[self.current_processing_requests][7]
                                                    * self.traffic_decreasing_ratio
                                                )
                                                self.flag += 1
                                                self.current_node = selected_neighbor
                                                reward = self.lev2
                                                return reward, terminate
                                            else:
                                                # print('----------bandwidth failed-----------')
                                                reward = self.lev0
                                                self.flag = 0
                                                self.new_request = True
                                                self.current_processing_requests += 1
                                                self.failure += 1
                                                return reward, terminate
                                        else:
                                            # print('----------computing satisfied-----------')
                                            reward = self.lev0
                                            self.flag = 0
                                            self.new_request = True
                                            self.current_processing_requests += 1
                                            self.failure += 1
                                            return reward, terminate
                                    else:
                                        # print('----------fronthaul failed-----------')
                                        reward = self.lev0
                                        self.flag = 0
                                        self.new_request = True
                                        self.current_processing_requests += 1
                                        self.failure += 1
                                        return reward, terminate
                                elif self.flag == 1:
                                    # print('----------no CUUP yet-----------')
                                    if (
                                        self.current_delay
                                        + self.network.node_list[self.current_node].get_distance(selected_neighbor)
                                        <= self.requests[self.current_processing_requests][2]
                                    ): 
                                        # print('----------midhaul satisfied-----------') 
                                        if (
                                            self.network.node_list[selected_neighbor].get_resource()
                                            >= self.requests[self.current_processing_requests][self.flag + 3]
                                        ):  
                                            # print('---------computing satisfied---------')
                                            if (
                                                self.network.node_list[self.current_node].get_bandwidth(
                                                    selected_neighbor
                                                )
                                                >= self.requests[self.current_processing_requests][7]
                                            ):  
                                                # print('----------bandwidth satisfied-----------')
                                                self.current_delay += self.network.node_list[
                                                    self.current_node
                                                ].get_distance(
                                                    selected_neighbor
                                                )  
                                                self.hop += 1  
                                                self.network.node_list[selected_neighbor].change_computing(
                                                    -self.requests[self.current_processing_requests][self.flag + 3]
                                                )  
                                                self.network.node_list[self.current_node].change_bandwidth(
                                                    selected_neighbor,
                                                    -self.requests[self.current_processing_requests][7],
                                                )  
                                                self.requests[self.current_processing_requests][7] = (
                                                    self.requests[self.current_processing_requests][7]
                                                    * self.traffic_decreasing_ratio
                                                )  
                                                self.flag += 1
                                                self.current_node = selected_neighbor
                                                reward = self.lev2
                                                return reward, terminate
                                            else:
                                                # print('----------bandwidth failed-----------')
                                                reward = self.lev0
                                                self.flag = 0
                                                self.new_request = True
                                                self.current_processing_requests += 1
                                                self.failure += 1
                                                return reward, terminate
                                        else:
                                            # print('----------computing failed-----------')
                                            reward = self.lev0
                                            self.flag = 0
                                            self.new_request = True
                                            self.current_processing_requests += 1
                                            self.failure += 1
                                            return reward, terminate
                                    else:
                                        # print('----------midhaul failed-----------')
                                        reward = self.lev0
                                        self.flag = 0
                                        self.new_request = True
                                        self.current_processing_requests += 1
                                        self.failure += 1
                                        return reward, terminate
                                elif self.flag == 2:
                                    if (
                                        self.current_delay
                                        + self.network.node_list[self.current_node].get_distance(selected_neighbor)
                                        <= self.requests[self.current_processing_requests][2]
                                    ): 
                                        # print('----------midhaul satisfied-----------')
                                        if (
                                            self.network.node_list[selected_neighbor].get_resource()
                                            >= self.requests[self.current_processing_requests][self.flag + 3]
                                        ):  
                                            # print('---------computing satisfied---------')
                                            if (
                                                self.network.node_list[self.current_node].get_bandwidth(
                                                    selected_neighbor
                                                )
                                                >= self.requests[self.current_processing_requests][7]
                                            ): 
                                                self.network.node_list[selected_neighbor].change_computing(
                                                    -self.requests[self.current_processing_requests][self.flag + 3]
                                                )  
                                                self.network.node_list[self.current_node].change_bandwidth(
                                                    selected_neighbor,
                                                    -self.requests[self.current_processing_requests][7],
                                                ) 
                                                self.requests[self.current_processing_requests][7] = (
                                                    self.requests[self.current_processing_requests][7]
                                                    * self.traffic_decreasing_ratio
                                                )  
                                                self.hop += 1 
                                                self.flag = 0  
                                                self.new_request = True
                                                self.current_processing_requests += 1
                                                self.current_node = selected_neighbor
                                                reward = self.lev3
                                                return reward, terminate
                                            else:
                                                # print('----------bandwidth failed-----------')
                                                reward = self.lev0
                                                self.flag = 0
                                                self.new_request = True
                                                self.current_processing_requests += 1
                                                self.failure += 1
                                                return reward, terminate
                                        else:
                                            # print('----------computing failed-----------')
                                            reward = self.lev0
                                            self.flag = 0
                                            self.new_request = True
                                            self.current_processing_requests += 1
                                            self.failure += 1
                                            return reward, terminate
                                    else:
                                        # print('----------midhaul failed-----------')
                                        reward = self.lev0
                                        self.flag = 0
                                        self.new_request = True
                                        self.current_processing_requests += 1
                                        self.failure += 1
                                        return reward, terminate

                            else:  
                                # print('----------no deploy-----------')
                                if self.flag == 0:  
                                    # print('----------no DU yet-----------')
                                    if (
                                        self.current_delay
                                        + self.network.node_list[self.current_node].get_distance(selected_neighbor)
                                        <= self.requests[self.current_processing_requests][1]
                                    ):  
                                        # print('----------fronthaul satisfied-----------')
                                        if (
                                            self.network.node_list[self.current_node].get_bandwidth(selected_neighbor)
                                            >= self.requests[self.current_processing_requests][7]
                                        ):  
                                            # print('----------bandwidth satisfied-----------')
                                            self.current_delay += self.network.node_list[
                                                self.current_node
                                            ].get_distance(
                                                selected_neighbor
                                            )  
                                            self.hop += 1  
                                            self.network.node_list[self.current_node].change_bandwidth(
                                                selected_neighbor, -self.requests[self.current_processing_requests][7]
                                            )  
                                            reward = self.lev1
                                            self.current_node = selected_neighbor
                                            return reward, terminate
                                        else:
                                            # print('----------bandwidth failed-----------')
                                            reward = self.lev0
                                            self.flag = 0
                                            self.new_request = True
                                            self.current_processing_requests += 1
                                            self.failure += 1
                                            return reward, terminate
                                    else:
                                        # print('----------fronthaul failed-----------')
                                        reward = self.lev0
                                        self.flag = 0
                                        self.new_request = True
                                        self.current_processing_requests += 1
                                        self.failure += 1
                                        return reward, terminate
                                else:  
                                    # print('----------DU done, CU, UPF now-----------')
                                    if (
                                        self.current_delay
                                        + self.network.node_list[self.current_node].get_distance(selected_neighbor)
                                        <= self.requests[self.current_processing_requests][2]
                                    ):  
                                        # print('----------midhaul satisfied-----------')
                                        if (
                                            self.network.node_list[self.current_node].get_bandwidth(selected_neighbor)
                                            >= self.requests[self.current_processing_requests][7]
                                        ):  
                                            # print('----------bandwidth satisfied-----------')
                                            self.current_delay += self.network.node_list[
                                                self.current_node
                                            ].get_distance(
                                                selected_neighbor
                                            )  
                                            self.hop += 1 
                                            self.network.node_list[self.current_node].change_bandwidth(
                                                selected_neighbor, -self.requests[self.current_processing_requests][7]
                                            )  
                                            reward = self.lev1
                                            self.current_node = selected_neighbor
                                            return reward, terminate
                                        else:
                                            # print('----------bandwidth failed-----------')
                                            reward = self.lev0
                                            self.flag = 0
                                            self.new_request = True
                                            self.current_processing_requests += 1
                                            self.failure += 1
                                            return reward, terminate
                                    else:
                                        # print('----------midhaul failed-----------')
                                        reward = self.lev0
                                        self.flag = 0
                                        self.new_request = True
                                        self.current_processing_requests += 1
                                        self.failure += 1
                                        return reward, terminate
                        else:  
                            # print('----------UN valid actoin-----------')
                            reward = self.lev0
                            self.flag = 0
                            self.new_request = True
                            self.current_processing_requests += 1
                            self.failure += 1
                            return reward, terminate
                    else: 
                        # print('----------neighbor_index == 4:-----------')
                        if self.flag == 0:
                            # print('----------no DU yet-----------')
                            if (
                                self.network.node_list[self.current_node].get_resource()
                                >= self.requests[self.current_processing_requests][self.flag + 3]
                                and self.network.node_list[self.current_node].node_type == 1
                            ): 
                                # print('----------computing satisfied-----------')
                                self.network.node_list[self.current_node].change_computing(
                                    -self.requests[self.current_processing_requests][self.flag + 3]
                                )  
                                self.requests[self.current_processing_requests][7] = (
                                    self.requests[self.current_processing_requests][7] * self.traffic_decreasing_ratio
                                )  # taffic大小减小
                                self.flag += 1
                                reward = self.lev2
                                return reward, terminate
                            else:
                                # print('----------computing failed-----------')
                                reward = self.lev0
                                self.flag = 0
                                self.new_request = True
                                self.current_processing_requests += 1
                                self.failure += 1
                                return reward, terminate
                        elif self.flag == 1:
                            # print('----------no CU yet-----------')
                            if (
                                self.network.node_list[self.current_node].get_resource()
                                >= self.requests[self.current_processing_requests][self.flag + 3]
                            ):  
                                # print('----------computing satisfied-----------')
                                self.network.node_list[self.current_node].change_computing(
                                    -self.requests[self.current_processing_requests][self.flag + 3]
                                )  
                                self.requests[self.current_processing_requests][7] = (
                                    self.requests[self.current_processing_requests][7] * self.traffic_decreasing_ratio
                                )  
                                self.flag += 1
                                reward = self.lev2
                                return reward, terminate
                            else:
                                # print('----------computing failed-----------')
                                reward = self.lev0
                                self.flag = 0
                                self.new_request = True
                                self.current_processing_requests += 1
                                self.failure += 1
                                return reward, terminate
                        elif self.flag == 2:
                            # print('----------no UPF yet-----------')
                            if (
                                self.network.node_list[self.current_node].get_resource()
                                >= self.requests[self.current_processing_requests][self.flag + 3]
                            ):  
                                # print('----------computing satisfied-----------')
                                self.network.node_list[self.current_node].change_computing(
                                    -self.requests[self.current_processing_requests][self.flag + 3]
                                )
                                self.requests[self.current_processing_requests][7] = (
                                    self.requests[self.current_processing_requests][7] * self.traffic_decreasing_ratio
                                )  
                                self.flag = 0  
                                self.new_request = True
                                self.current_processing_requests += 1
                                reward = self.lev3
                                return reward, terminate
                            else:  
                                reward = self.lev0
                                self.flag = 0
                                self.new_request = True
                                self.current_processing_requests += 1
                                self.failure += 1
                                return reward, terminate
            else:  
                # print('--------------the last_effect_request--------------')
                terminate = True
                neighbor_index = action // 2
                if (
                    self.requests[self.current_processing_requests][0] == 0
                ):  
                    # print('-----------uRLLC------------')
                    if neighbor_index != 4:  
                        # print('-----------neighbor_index != 4------------')
                        if neighbor_index < len(
                            self.network.node_list[self.current_node].get_neighbors()
                        ):  
                            # print('-----------valid action------------')
                            selected_neighbor = list(self.network.node_list[self.current_node].get_neighbors())[
                                neighbor_index
                            ]
                            deployment_decision = action % 2 

                         
                            if deployment_decision == 1: 
                                # print('-----------deploy------------')
                                if self.flag == 0:  
                                    # print('-----------no DU yet------------')
                                    if (
                                        self.current_delay
                                        + self.network.node_list[self.current_node].get_distance(selected_neighbor)
                                        <= self.requests[self.current_processing_requests][1]
                                    ):  
                                        # print('-----------fronthaul satified------------')
                                        if (
                                            self.network.node_list[selected_neighbor].get_resource()
                                            >= self.requests[self.current_processing_requests][self.flag + 3]
                                            and self.network.node_list[selected_neighbor].node_type == 1
                                        ):  
                                            # print('-----------computing satisfied------------')
                                            if (
                                                self.network.node_list[self.current_node].get_bandwidth(
                                                    selected_neighbor
                                                )
                                                >= self.requests[self.current_processing_requests][7]
                                            ):  
                                                # print('-----------bandwidth satisfied------------')
                                                self.current_delay += self.network.node_list[
                                                    self.current_node
                                                ].get_distance(
                                                    selected_neighbor
                                                )
                                                self.hop += 1
                                                self.network.node_list[selected_neighbor].change_computing(
                                                    -self.requests[self.current_processing_requests][self.flag + 3]
                                                )  
                                                self.network.node_list[self.current_node].change_bandwidth(
                                                    selected_neighbor,
                                                    -self.requests[self.current_processing_requests][7],
                                                )  
                                                self.requests[self.current_processing_requests][7] = (
                                                    self.requests[self.current_processing_requests][7]
                                                    * self.traffic_decreasing_ratio
                                                )  
                                                self.current_node = selected_neighbor
                                                self.flag += 1
                                                reward = self.lev2
                                                terminate = False
                                                return reward, terminate
                                            else:
                                                # print('-----------bandwidth failed------------')
                                                self.failure += 1
                                                self.success_ratio = 1 - self.failure / self.total_effect_request
                                                if self.success_ratio <= 1 / 2:
                                                    reward = self.lev4_0
                                                elif 1 / 2 < self.success_ratio:
                                                    reward = self.lev4_1 * self.success_ratio
                                                self.episode.append(reward)
                                                return reward, terminate
                                        else:
                                            # print('-----------computing failed------------')
                                            self.failure += 1
                                            self.success_ratio = 1 - self.failure / self.total_effect_request
                                            if self.success_ratio <= 1 / 2:
                                                reward = self.lev4_0
                                            elif 1 / 2 < self.success_ratio:
                                                reward = self.lev4_1 * self.success_ratio
                                            self.episode.append(reward)
                                            return reward, terminate
                                    else:
                                        # print('-----------fronthaul failed------------')
                                        self.failure += 1
                                        self.success_ratio = 1 - self.failure / self.total_effect_request
                                        if self.success_ratio <= 1 / 2:
                                            reward = self.lev4_0
                                        elif 1 / 2 < self.success_ratio:
                                            reward = self.lev4_1 * self.success_ratio
                                        self.episode.append(reward)
                                        return reward, terminate
                                else:  
                                    # print('-----------DU done, CU UPF now------------')
                                    if (
                                        self.current_delay
                                        + self.network.node_list[self.current_node].get_distance(selected_neighbor)
                                        <= self.requests[self.current_processing_requests][2]
                                    ):  
                                        # print('-----------e2e satified------------')
                                        if (
                                            self.network.node_list[selected_neighbor].get_resource()
                                            >= self.requests[self.current_processing_requests][self.flag + 3]
                                        ):  
                                            # print('-----------computing satisfied------------')
                                            if (
                                                self.network.node_list[self.current_node].get_bandwidth(
                                                    selected_neighbor
                                                )
                                                >= self.requests[self.current_processing_requests][7]
                                            ):  
                                                # print('-----------bandwidth satisfied------------')
                                                self.current_delay += self.network.node_list[
                                                    self.current_node
                                                ].get_distance(
                                                    selected_neighbor
                                                )  
                                                self.hop += 1 
                                                self.network.node_list[selected_neighbor].change_computing(
                                                    -self.requests[self.current_processing_requests][self.flag + 3]
                                                )  
                                                self.network.node_list[self.current_node].change_bandwidth(
                                                    selected_neighbor,
                                                    -self.requests[self.current_processing_requests][7],
                                                )  
                                                self.requests[self.current_processing_requests][7] = (
                                                    self.requests[self.current_processing_requests][7]
                                                    * self.traffic_decreasing_ratio
                                                )  
                                                self.flag += 1
                                                self.current_node = selected_neighbor
                                                if self.flag == 4 and self.failure == 0:
                                                    # print('-----------self.flag == 4 and self.failure == 0------------')
                                                    total_power = 0
                                                    for node in list(self.network.get_node_list())[:-1]:
                                                        power = calculate_power(node)
                                                        total_power += power
                                                    total_power += self.hop * self.unified_switching_power
                                                    average_power = total_power / self.total_effect_request
                                                    reward = self.lev4_2
                                                    if self.network_index == 0:
                                                        store_data(
                                                            self.power_recorder_0, average_power, self.slicing_window
                                                        )
                                                        store_data(
                                                            self.power_recorder_0_all, average_power, 500
                                                        )
                                                        if len(self.power_recorder_0) < self.slicing_window:
                                                            reward += (
                                                                500 / average_power
                                                            )  
                                                        else:
                                                            reward += (
                                                                bottom_avg(self.power_recorder_0) / average_power
                                                            )
                                                    elif self.network_index == 1:
                                                        store_data(
                                                            self.power_recorder_1, average_power, self.slicing_window
                                                        )
                                                        store_data(
                                                            self.power_recorder_1_all, average_power, 500
                                                        )
                                                        if len(self.power_recorder_1) < self.slicing_window:
                                                            reward += (
                                                                500 / average_power
                                                            )  
                                                        else:
                                                            reward += (
                                                                bottom_avg(self.power_recorder_1) / average_power
                                                            )
                                                    elif self.network_index == 2:
                                                        store_data(
                                                            self.power_recorder_2, average_power, self.slicing_window
                                                        )
                                                        store_data(
                                                            self.power_recorder_2_all, average_power, 500
                                                        )
                                                        if len(self.power_recorder_2) < self.slicing_window:
                                                            reward += 500 / average_power
                                                        else:
                                                            reward += (
                                                                bottom_avg(self.power_recorder_2) / average_power
                                                            )
                                                    elif self.network_index == 3:
                                                        store_data(
                                                            self.power_recorder_3, average_power, self.slicing_window
                                                        )
                                                        store_data(
                                                            self.power_recorder_3_all, average_power, 500
                                                        )
                                                        if len(self.power_recorder_3) < self.slicing_window:
                                                            reward += 500 / average_power
                                                        else:
                                                            reward += (
                                                                bottom_avg(self.power_recorder_3) / average_power
                                                            )
                                                    self.episode.append(reward)
                                                    return reward, terminate
                                                elif self.flag == 4 and self.failure != 0:
                                                    # print('-----------self.flag == 4 and self.failure != 0------------')
                                                    self.success_ratio = 1 - self.failure / self.total_effect_request
                                                    if self.success_ratio <= 1 / 2:
                                                        reward = self.lev4_0
                                                    elif 1 / 2 < self.success_ratio:
                                                        reward = self.lev4_1 * self.success_ratio
                                                    self.episode.append(reward)
                                                    return reward, terminate
                                                else: 
                                                    reward = self.lev2
                                                    terminate = False
                                                    return reward, terminate
                                            else:
                                                # print('-----------bandwidth failed------------')
                                                self.failure += 1
                                                self.success_ratio = 1 - self.failure / self.total_effect_request
                                                if self.success_ratio <= 1 / 2:
                                                    reward = self.lev4_0
                                                elif 1 / 2 < self.success_ratio:
                                                    reward = self.lev4_1 * self.success_ratio
                                                self.episode.append(reward)
                                                return reward, terminate
                                        else:
                                            # print('-----------computing failed------------')
                                            self.failure += 1
                                            self.success_ratio = 1 - self.failure / self.total_effect_request
                                            if self.success_ratio <= 1 / 2:
                                                reward = self.lev4_0
                                            elif 1 / 2 < self.success_ratio:
                                                reward = self.lev4_1 * self.success_ratio
                                            self.episode.append(reward)
                                            return reward, terminate
                                    else:
                                        # print('-----------e2e failed------------')
                                        self.failure += 1
                                        self.success_ratio = 1 - self.failure / self.total_effect_request
                                        if self.success_ratio <= 1 / 2:
                                            reward = self.lev4_0
                                        elif 1 / 2 < self.success_ratio:
                                            reward = self.lev4_1 * self.success_ratio
                                        self.episode.append(reward)
                                        return reward, terminate
                            else:  
                                # print('-----------no deployment------------')
                                if self.flag == 0:  
                                    # print('-----------no DU yet------------')
                                    if (
                                        self.current_delay
                                        + self.network.node_list[self.current_node].get_distance(selected_neighbor)
                                        <= self.requests[self.current_processing_requests][1]
                                    ):  
                                        # print('-----------fronthaul satisfied------------')
                                        if (
                                            self.network.node_list[self.current_node].get_bandwidth(selected_neighbor)
                                            >= self.requests[self.current_processing_requests][7]
                                        ):  
                                            # print('-----------bandwidth satisfied------------')
                                            self.current_delay += self.network.node_list[
                                                self.current_node
                                            ].get_distance(
                                                selected_neighbor
                                            )  
                                            self.hop += 1  
                                            self.network.node_list[self.current_node].change_bandwidth(
                                                selected_neighbor, -self.requests[self.current_processing_requests][7]
                                            ) 
                                            reward = self.lev1
                                            self.current_node = selected_neighbor
                                            terminate = False
                                            return reward, terminate
                                        else:
                                            # print('-----------bandwidth failed------------')
                                            self.failure += 1
                                            self.success_ratio = 1 - self.failure / self.total_effect_request
                                            if self.success_ratio <= 1 / 2:
                                                reward = self.lev4_0
                                            elif 1 / 2 < self.success_ratio:
                                                reward = self.lev4_1 * self.success_ratio
                                            self.episode.append(reward)
                                            return reward, terminate
                                    else:
                                        # print('-----------fronthaul failed------------')
                                        self.failure += 1
                                        self.success_ratio = 1 - self.failure / self.total_effect_request
                                        if self.success_ratio <= 1 / 2:
                                            reward = self.lev4_0
                                        elif 1 / 2 < self.success_ratio:
                                            reward = self.lev4_1 * self.success_ratio
                                        self.episode.append(reward)
                                        return reward, terminate
                                else:  
                                    # print('-----------DU done, CU UPF now------------')
                                    if (
                                        self.current_delay
                                        + self.network.node_list[self.current_node].get_distance(selected_neighbor)
                                        <= self.requests[self.current_processing_requests][2]
                                    ):  
                                        # print('-----------e2e satisfied------------')
                                        if (
                                            self.network.node_list[self.current_node].get_bandwidth(selected_neighbor)
                                            >= self.requests[self.current_processing_requests][7]
                                        ):  
                                            # print('-----------bandwidth satisfied------------')
                                            self.current_delay += self.network.node_list[
                                                self.current_node
                                            ].get_distance(
                                                selected_neighbor
                                            )  
                                            self.hop += 1  
                                            self.network.node_list[self.current_node].change_bandwidth(
                                                selected_neighbor, -self.requests[self.current_processing_requests][7]
                                            ) 
                                            reward = self.lev1
                                            self.current_node = selected_neighbor
                                            terminate = False
                                            return reward, terminate
                                        else:
                                            # print('-----------bandwidth failed------------')
                                            self.failure += 1
                                            self.success_ratio = 1 - self.failure / self.total_effect_request
                                            if self.success_ratio <= 1 / 2:
                                                reward = self.lev4_0
                                            elif 1 / 2 < self.success_ratio:
                                                reward = self.lev4_1 * self.success_ratio
                                            self.episode.append(reward)
                                            return reward, terminate
                                    else:
                                        # print('-----------e2e failed------------')
                                        self.failure += 1
                                        self.success_ratio = 1 - self.failure / self.total_effect_request
                                        if self.success_ratio <= 1 / 2:
                                            reward = self.lev4_0
                                        elif 1 / 2 < self.success_ratio:
                                            reward = self.lev4_1 * self.success_ratio
                                        self.episode.append(reward)
                                        return reward, terminate
                        else: 
                            # print('-----------UN valid action------------')
                            self.failure += 1
                            self.success_ratio = 1 - self.failure / self.total_effect_request
                            if self.success_ratio <= 1 / 2:
                                reward = self.lev4_0
                            elif 1 / 2 < self.success_ratio:
                                reward = self.lev4_1 * self.success_ratio
                            self.episode.append(reward)
                            return reward, terminate
                    else:  
                        # print('-----------last action index------------')
                        if self.flag == 0:  
                            # print('-----------no DU yet------------')
                            if (
                                self.network.node_list[self.current_node].get_resource()
                                >= self.requests[self.current_processing_requests][self.flag + 3]
                                and self.network.node_list[self.current_node].node_type == 1
                            ): 
                                # print('-----------computing satisfied------------')
                                self.network.node_list[self.current_node].change_computing(
                                    -self.requests[self.current_processing_requests][self.flag + 3]
                                )  
                                self.requests[self.current_processing_requests][7] = (
                                    self.requests[self.current_processing_requests][7] * self.traffic_decreasing_ratio
                                )  
                                self.flag += 1
                                reward = self.lev2
                                terminate = False
                                return reward, terminate
                            else:
                                # print('-----------computing failed------------')
                                self.failure += 1
                                self.success_ratio = 1 - self.failure / self.total_effect_request
                                if self.success_ratio <= 1 / 2:
                                    reward = self.lev4_0
                                elif 1 / 2 < self.success_ratio:
                                    reward = self.lev4_1 * self.success_ratio
                                self.episode.append(reward)
                                return reward, terminate
                        else: 
                            # print('-----------DU done, CU UPF now------------')
                            if (
                                self.network.node_list[self.current_node].get_resource()
                                >= self.requests[self.current_processing_requests][self.flag + 3]
                            ):  
                                # print('-----------computing satisfied------------')
                                self.network.node_list[self.current_node].change_computing(
                                    -self.requests[self.current_processing_requests][self.flag + 3]
                                )  
                                self.requests[self.current_processing_requests][7] = (
                                    self.requests[self.current_processing_requests][7] * self.traffic_decreasing_ratio
                                )  # taffic大小减小
                                self.flag += 1
                                if self.flag == 4 and self.failure == 0:
                                    total_power = 0
                                    for node in list(self.network.get_node_list())[:-1]:
                                        power = calculate_power(node)
                                        total_power += power
                                    total_power += self.hop * self.unified_switching_power
                                    average_power = total_power / self.total_effect_request
                                    reward = self.lev4_2
                                    if self.network_index == 0:
                                        store_data(self.power_recorder_0, average_power, self.slicing_window)
                                        store_data(self.power_recorder_0_all, average_power, 500)
                                        if len(self.power_recorder_0) < self.slicing_window:
                                            reward += 500 / average_power
                                        else:
                                            reward += (
                                                bottom_avg(self.power_recorder_0) / average_power
                                            )
                                    elif self.network_index == 1:
                                        store_data(self.power_recorder_1, average_power, self.slicing_window)
                                        store_data(self.power_recorder_1_all, average_power, 500)
                                        if len(self.power_recorder_1) < self.slicing_window:
                                            reward += 500 / average_power
                                        else:
                                            reward += (
                                                bottom_avg(self.power_recorder_1) / average_power
                                            )
                                    elif self.network_index == 2:
                                        store_data(self.power_recorder_2, average_power, self.slicing_window)
                                        store_data(self.power_recorder_2_all, average_power, 500)
                                        if len(self.power_recorder_2) < self.slicing_window:
                                            reward += 500 / average_power
                                        else:
                                            reward += (
                                                bottom_avg(self.power_recorder_2) / average_power
                                            )
                                    elif self.network_index == 3:
                                        store_data(self.power_recorder_3, average_power, self.slicing_window)
                                        store_data(self.power_recorder_3_all, average_power, 500)
                                        if len(self.power_recorder_3) < self.slicing_window:
                                            reward += 500 / average_power
                                        else:
                                            reward += (
                                                bottom_avg(self.power_recorder_3) / average_power
                                            )
                                    self.episode.append(reward)
                                    return reward, terminate
                                elif self.flag == 4 and self.failure != 0:
                                    self.success_ratio = 1 - self.failure / self.total_effect_request
                                    if self.success_ratio <= 1 / 2:
                                        reward = self.lev4_0
                                    elif 1 / 2 < self.success_ratio:
                                        reward = self.lev4_1 * self.success_ratio
                                    self.episode.append(reward)
                                    return reward, terminate
                                else:  # self.flag = 1, 2, 3
                                    reward = self.lev2
                                    terminate = False
                                    return reward, terminate
                            else:
                                # print('-----------computing failed------------')
                                self.failure += 1
                                self.success_ratio = 1 - self.failure / self.total_effect_request
                                if self.success_ratio <= 1 / 2:
                                    reward = self.lev4_0
                                elif 1 / 2 < self.success_ratio:
                                    reward = self.lev4_1 * self.success_ratio
                                self.episode.append(reward)
                                return reward, terminate
                else:  
                    # self.destination
                    # print('-----------eMBB & mMTC------------')
                    if neighbor_index != 4:  
                        # print('-----------neighbor_index != 4------------')
                        if neighbor_index < len(
                            self.network.node_list[self.current_node].get_neighbors()
                        ):  
                            # print('-----------valid action------------')
                            selected_neighbor = list(self.network.node_list[self.current_node].get_neighbors())[
                                neighbor_index
                            ]
                            deployment_decision = action % 2  

                          
                            if deployment_decision == 1: 
                                if self.flag == 0: 
                                    if (
                                        self.current_delay
                                        + self.network.node_list[self.current_node].get_distance(selected_neighbor)
                                        <= self.requests[self.current_processing_requests][1]
                                    ):  
                                        if (
                                            self.network.node_list[selected_neighbor].get_resource()
                                            >= self.requests[self.current_processing_requests][self.flag + 3]
                                            and self.network.node_list[selected_neighbor].node_type == 1
                                        ):  
                                            if (
                                                self.network.node_list[self.current_node].get_bandwidth(
                                                    selected_neighbor
                                                )
                                                >= self.requests[self.current_processing_requests][7]
                                            ):  
                                                self.current_delay += self.network.node_list[
                                                    self.current_node
                                                ].get_distance(
                                                    selected_neighbor
                                                )  
                                                self.hop += 1  
                                                self.network.node_list[selected_neighbor].change_computing(
                                                    -self.requests[self.current_processing_requests][self.flag + 3]
                                                )  
                                                self.network.node_list[self.current_node].change_bandwidth(
                                                    selected_neighbor,
                                                    -self.requests[self.current_processing_requests][7],
                                                ) 
                                                self.requests[self.current_processing_requests][7] = (
                                                    self.requests[self.current_processing_requests][7]
                                                    * self.traffic_decreasing_ratio
                                                ) 
                                                self.flag += 1
                                                reward = self.lev2
                                                self.current_node = selected_neighbor
                                                terminate = False
                                                return reward, terminate
                                            else:
                                                self.failure += 1
                                                self.success_ratio = 1 - self.failure / self.total_effect_request
                                                if self.success_ratio <= 1 / 2:
                                                    reward = self.lev4_0
                                                elif 1 / 2 < self.success_ratio:
                                                    reward = self.lev4_1 * self.success_ratio
                                                self.episode.append(reward)
                                                return reward, terminate
                                        else:
                                            self.failure += 1
                                            self.success_ratio = 1 - self.failure / self.total_effect_request
                                            if self.success_ratio <= 1 / 2:
                                                reward = self.lev4_0
                                            elif 1 / 2 < self.success_ratio:
                                                reward = self.lev4_1 * self.success_ratio
                                            self.episode.append(reward)
                                            return reward, terminate
                                    else:
                                        self.failure += 1
                                        self.success_ratio = 1 - self.failure / self.total_effect_request
                                        if self.success_ratio <= 1 / 2:
                                            reward = self.lev4_0
                                        elif 1 / 2 < self.success_ratio:
                                            reward = self.lev4_1 * self.success_ratio
                                        self.episode.append(reward)
                                        return reward, terminate
                                elif self.flag == 1:
                                    if (
                                        self.current_delay
                                        + self.network.node_list[self.current_node].get_distance(selected_neighbor)
                                        <= self.requests[self.current_processing_requests][2]
                                    ):  
                                        if (
                                            self.network.node_list[selected_neighbor].get_resource()
                                            >= self.requests[self.current_processing_requests][self.flag + 3]
                                        ): 
                                            if (
                                                self.network.node_list[self.current_node].get_bandwidth(
                                                    selected_neighbor
                                                )
                                                >= self.requests[self.current_processing_requests][7]
                                            ):  
                                                self.current_delay += self.network.node_list[
                                                    self.current_node
                                                ].get_distance(
                                                    selected_neighbor
                                                ) 
                                                self.hop += 1  
                                                self.network.node_list[selected_neighbor].change_computing(
                                                    -self.requests[self.current_processing_requests][self.flag + 3]
                                                )  
                                                self.network.node_list[self.current_node].change_bandwidth(
                                                    selected_neighbor,
                                                    -self.requests[self.current_processing_requests][7],
                                                )
                                                self.requests[self.current_processing_requests][7] = (
                                                    self.requests[self.current_processing_requests][7]
                                                    * self.traffic_decreasing_ratio
                                                ) 
                                                self.flag += 1
                                                reward = self.lev2
                                                self.current_node = selected_neighbor
                                                terminate = False
                                                return reward, terminate
                                            else:
                                                self.failure += 1
                                                self.success_ratio = 1 - self.failure / self.total_effect_request
                                                if self.success_ratio <= 1 / 2:
                                                    reward = self.lev4_0
                                                elif 1 / 2 < self.success_ratio:
                                                    reward = self.lev4_1 * self.success_ratio
                                                self.episode.append(reward)
                                                return reward, terminate
                                        else:
                                            self.failure += 1
                                            self.success_ratio = 1 - self.failure / self.total_effect_request
                                            if self.success_ratio <= 1 / 2:
                                                reward = self.lev4_0
                                            elif 1 / 2 < self.success_ratio:
                                                reward = self.lev4_1 * self.success_ratio
                                            self.episode.append(reward)
                                            return reward, terminate
                                    else:
                                        self.failure += 1
                                        self.success_ratio = 1 - self.failure / self.total_effect_request
                                        if self.success_ratio <= 1 / 2:
                                            reward = self.lev4_0
                                        elif 1 / 2 < self.success_ratio:
                                            reward = self.lev4_1 * self.success_ratio
                                        self.episode.append(reward)
                                        return reward, terminate
                                elif self.flag == 2:
                                    if (
                                        self.current_delay
                                        + self.network.node_list[self.current_node].get_distance(selected_neighbor)
                                        <= self.requests[self.current_processing_requests][2]
                                    ):  
                                        if (
                                            self.network.node_list[selected_neighbor].get_resource()
                                            >= self.requests[self.current_processing_requests][self.flag + 3]
                                        ):  
                                            if (
                                                self.network.node_list[self.current_node].get_bandwidth(
                                                    selected_neighbor
                                                )
                                                >= self.requests[self.current_processing_requests][7]
                                            ):  
                                                self.current_delay += self.network.node_list[
                                                    self.current_node
                                                ].get_distance(
                                                    selected_neighbor
                                                )  
                                                self.hop += 1
                                                self.network.node_list[selected_neighbor].change_computing(
                                                    -self.requests[self.current_processing_requests][self.flag + 3]
                                                )  
                                                self.network.node_list[self.current_node].change_bandwidth(
                                                    selected_neighbor,
                                                    -self.requests[self.current_processing_requests][7],
                                                )  
                                                self.requests[self.current_processing_requests][7] = (
                                                    self.requests[self.current_processing_requests][7]
                                                    * self.traffic_decreasing_ratio
                                                )  

                                                self.flag = 0  
                                                self.current_node = selected_neighbor
                                                self.success_ratio = 1 - self.failure / self.total_effect_request
                                                if self.success_ratio <= 1 / 2:
                                                    reward = self.lev4_0
                                                elif 1 / 2 < self.success_ratio < 1:
                                                    reward = self.lev4_1 * self.success_ratio
                                                elif self.success_ratio == 1:
                                                    total_power = 0
                                                    for node in list(self.network.get_node_list())[:-1]:
                                                        power = calculate_power(node)
                                                        total_power += power
                                                    total_power += self.hop * self.unified_switching_power
                                                    average_power = total_power / self.total_effect_request
                                                    reward = self.lev4_2
                                                    if self.network_index == 0:
                                                        store_data(
                                                            self.power_recorder_0, average_power, self.slicing_window
                                                        )
                                                        store_data(self.power_recorder_0_all, average_power, 500)
                                                        if len(self.power_recorder_0) < self.slicing_window:
                                                            reward += 500 / average_power
                                                        else:
                                                            reward += (
                                                                bottom_avg(self.power_recorder_0) / average_power
                                                            )
                                                    elif self.network_index == 1:
                                                        store_data(
                                                            self.power_recorder_1, average_power, self.slicing_window
                                                        )
                                                        store_data(self.power_recorder_1_all, average_power, 500)
                                                        if len(self.power_recorder_1) < self.slicing_window:
                                                            reward += 500 / average_power 
                                                        else:
                                                            reward += (
                                                                bottom_avg(self.power_recorder_1) / average_power
                                                            )
                                                    elif self.network_index == 2:
                                                        store_data(
                                                            self.power_recorder_2, average_power, self.slicing_window
                                                        )
                                                        store_data(self.power_recorder_2_all, average_power, 500)
                                                        if len(self.power_recorder_2) < self.slicing_window:
                                                            reward += 500 / average_power 
                                                        else:
                                                            reward += (
                                                                bottom_avg(self.power_recorder_2) / average_power
                                                            )
                                                    elif self.network_index == 3:
                                                        store_data(
                                                            self.power_recorder_3, average_power, self.slicing_window
                                                        )
                                                        store_data(self.power_recorder_3_all, average_power, 500)
                                                        if len(self.power_recorder_3) < self.slicing_window:
                                                            reward += 500 / average_power
                                                        else:
                                                            reward += (
                                                                bottom_avg(self.power_recorder_3) / average_power
                                                            )
                                                self.episode.append(reward)
                                                return reward, terminate
                                            else:
                                                self.failure += 1
                                                self.success_ratio = 1 - self.failure / self.total_effect_request
                                                if self.success_ratio <= 1 / 2:
                                                    reward = self.lev4_0
                                                elif 1 / 2 < self.success_ratio:
                                                    reward = self.lev4_1 * self.success_ratio
                                                self.episode.append(reward)
                                                return reward, terminate
                                        else:
                                            self.failure += 1
                                            self.success_ratio = 1 - self.failure / self.total_effect_request
                                            if self.success_ratio <= 1 / 2:
                                                reward = self.lev4_0
                                            elif 1 / 2 < self.success_ratio:
                                                reward = self.lev4_1 * self.success_ratio
                                            self.episode.append(reward)
                                            return reward, terminate
                                    else:
                                        self.failure += 1
                                        self.success_ratio = 1 - self.failure / self.total_effect_request
                                        if self.success_ratio <= 1 / 2:
                                            reward = self.lev4_0
                                        elif 1 / 2 < self.success_ratio:
                                            reward = self.lev4_1 * self.success_ratio
                                        self.episode.append(reward)
                                        return reward, terminate
                            else: 
                                if self.flag == 0:  
                                    if (
                                        self.current_delay
                                        + self.network.node_list[self.current_node].get_distance(selected_neighbor)
                                        <= self.requests[self.current_processing_requests][1]
                                    ):  
                                        if (
                                            self.network.node_list[self.current_node].get_bandwidth(selected_neighbor)
                                            >= self.requests[self.current_processing_requests][7]
                                        ):  
                                            self.current_delay += self.network.node_list[
                                                self.current_node
                                            ].get_distance(
                                                selected_neighbor
                                            )  
                                            self.hop += 1  
                                            self.network.node_list[self.current_node].change_bandwidth(
                                                selected_neighbor, -self.requests[self.current_processing_requests][7]
                                            )  
                                            self.current_node = selected_neighbor
                                            reward = self.lev1
                                            terminate = False
                                            return reward, terminate
                                        else:
                                            self.failure += 1
                                            self.success_ratio = 1 - self.failure / self.total_effect_request
                                            if self.success_ratio <= 1 / 2:
                                                reward = self.lev4_0
                                            elif 1 / 2 < self.success_ratio:
                                                reward = self.lev4_1 * self.success_ratio
                                            self.episode.append(reward)
                                            return reward, terminate
                                    else:
                                        self.failure += 1
                                        self.success_ratio = 1 - self.failure / self.total_effect_request
                                        if self.success_ratio <= 1 / 2:
                                            reward = self.lev4_0
                                        elif 1 / 2 < self.success_ratio:
                                            reward = self.lev4_1 * self.success_ratio
                                        self.episode.append(reward)
                                        return reward, terminate
                                else: 
                                    if (
                                        self.current_delay
                                        + self.network.node_list[self.current_node].get_distance(selected_neighbor)
                                        <= self.requests[self.current_processing_requests][2]
                                    ): 
                                        if (
                                            self.network.node_list[self.current_node].get_bandwidth(selected_neighbor)
                                            >= self.requests[self.current_processing_requests][7]
                                        ):  
                                            self.current_delay += self.network.node_list[
                                                self.current_node
                                            ].get_distance(
                                                selected_neighbor
                                            )  
                                            self.hop += 1  
                                            self.network.node_list[self.current_node].change_bandwidth(
                                                selected_neighbor, -self.requests[self.current_processing_requests][7]
                                            )  
                                            self.current_node = selected_neighbor
                                            reward = self.lev1
                                            terminate = False
                                            return reward, terminate
                                        else:
                                            self.failure += 1
                                            self.success_ratio = 1 - self.failure / self.total_effect_request
                                            if self.success_ratio <= 1 / 2:
                                                reward = self.lev4_0
                                            elif 1 / 2 < self.success_ratio:
                                                reward = self.lev4_1 * self.success_ratio
                                            self.episode.append(reward)
                                            return reward, terminate
                                    else:
                                        self.failure += 1
                                        self.success_ratio = 1 - self.failure / self.total_effect_request
                                        if self.success_ratio <= 1 / 2:
                                            reward = self.lev4_0
                                        elif 1 / 2 < self.success_ratio:
                                            reward = self.lev4_1 * self.success_ratio
                                        self.episode.append(reward)
                                        return reward, terminate
                        else: 
                            # print('-----------UN valid action------------')
                            self.failure += 1
                            self.success_ratio = 1 - self.failure / self.total_effect_request
                            if self.success_ratio <= 1 / 2:
                                reward = self.lev4_0
                            elif 1 / 2 < self.success_ratio:
                                reward = self.lev4_1 * self.success_ratio
                            self.episode.append(reward)
                            return reward, terminate
                    else: 
                        # print('-----------neighbor_index == 4------------')
                        if self.flag == 0:  
                            if (
                                self.network.node_list[self.current_node].get_resource()
                                >= self.requests[self.current_processing_requests][self.flag + 3]
                                and self.network.node_list[self.current_node].node_type == 1
                            ):  
                                self.network.node_list[self.current_node].change_computing(
                                    -self.requests[self.current_processing_requests][self.flag + 3]
                                )  
                                self.requests[self.current_processing_requests][7] = (
                                    self.requests[self.current_processing_requests][7] * self.traffic_decreasing_ratio
                                )  
                                self.flag += 1
                                reward = self.lev2
                                terminate = False
                                return reward, terminate
                            else:
                                self.failure += 1
                                self.success_ratio = 1 - self.failure / self.total_effect_request
                                if self.success_ratio <= 1 / 2:
                                    reward = self.lev4_0
                                elif 1 / 2 < self.success_ratio:
                                    reward = self.lev4_1 * self.success_ratio
                                self.episode.append(reward)
                                return reward, terminate
                        elif self.flag == 1:
                            if (
                                self.network.node_list[self.current_node].get_resource()
                                >= self.requests[self.current_processing_requests][self.flag + 3]
                            ): 
                                self.network.node_list[self.current_node].change_computing(
                                    -self.requests[self.current_processing_requests][self.flag + 3]
                                ) 
                                self.requests[self.current_processing_requests][7] = (
                                    self.requests[self.current_processing_requests][7] * self.traffic_decreasing_ratio
                                )  
                                self.flag += 1
                                reward = self.lev2
                                terminate = False
                                return reward, terminate
                            else:
                                self.failure += 1
                                self.success_ratio = 1 - self.failure / self.total_effect_request
                                if self.success_ratio <= 1 / 2:
                                    reward = self.lev4_0
                                elif 1 / 2 < self.success_ratio:
                                    reward = self.lev4_1 * self.success_ratio
                                self.episode.append(reward)
                                return reward, terminate

                        elif self.flag == 2:
                            if (
                                self.network.node_list[self.current_node].get_resource()
                                >= self.requests[self.current_processing_requests][self.flag + 3]
                            ):  
                                self.network.node_list[self.current_node].change_computing(
                                    -self.requests[self.current_processing_requests][self.flag + 3]
                                )  
                                self.requests[self.current_processing_requests][7] = (
                                    self.requests[self.current_processing_requests][7] * self.traffic_decreasing_ratio
                                )  
                                self.flag = 0 

                                self.success_ratio = 1 - self.failure / self.total_effect_request
                                if self.success_ratio <= 1 / 2:
                                    reward = self.lev4_0
                                elif 1 / 2 < self.success_ratio < 1:
                                    reward = self.lev4_1 * self.success_ratio
                                elif self.success_ratio == 1:
                                    total_power = 0
                                    for node in list(self.network.get_node_list())[:-1]:
                                        power = calculate_power(node)
                                        total_power += power
                                    total_power += self.hop * self.unified_switching_power
                                    average_power = total_power / self.total_effect_request
                                    reward = self.lev4_2
                                    if self.network_index == 0:
                                        store_data(self.power_recorder_0, average_power, self.slicing_window)
                                        store_data(self.power_recorder_0_all, average_power, 500)
                                        if len(self.power_recorder_0) < self.slicing_window:
                                            reward += 500 / average_power
                                        else:
                                            reward += (
                                                bottom_avg(self.power_recorder_0) / average_power
                                            )
                                    elif self.network_index == 1:
                                        store_data(self.power_recorder_1, average_power, self.slicing_window)
                                        store_data(self.power_recorder_1_all, average_power, 500)
                                        if len(self.power_recorder_1) < self.slicing_window:
                                            reward += 500 / average_power
                                        else:
                                            reward += (
                                                bottom_avg(self.power_recorder_1) / average_power
                                            )
                                    elif self.network_index == 2:
                                        store_data(self.power_recorder_2, average_power, self.slicing_window)
                                        store_data(self.power_recorder_2_all, average_power, 500)
                                        if len(self.power_recorder_2) < self.slicing_window:
                                            reward += 500 / average_power
                                        else:
                                            reward += (
                                                bottom_avg(self.power_recorder_2) / average_power
                                            )
                                    elif self.network_index == 3:
                                        store_data(self.power_recorder_3, average_power, self.slicing_window)
                                        store_data(self.power_recorder_3_all, average_power, 500)
                                        if len(self.power_recorder_3) < self.slicing_window:
                                            reward += 500 / average_power
                                        else:
                                            reward += (
                                                bottom_avg(self.power_recorder_3) / average_power
                                            )
                                self.episode.append(reward)
                                return reward, terminate
                            else:
                                self.failure += 1
                                self.success_ratio = 1 - self.failure / self.total_effect_request
                                if self.success_ratio <= 1 / 2:
                                    reward = self.lev4_0
                                elif 1 / 2 < self.success_ratio:
                                    reward = self.lev4_1 * self.success_ratio
                                self.episode.append(reward)
                                return reward, terminate

    # DONE
    def reset(self):
        # initialize self.network, self.node_status, self.link_status
        self.build_network()
        self.generate_requests()
        self.ReadNet()

        return self.state.detach().numpy().astype(np.float32)
