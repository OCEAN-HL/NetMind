# NetMind

This work has been accepted by IEEE Wireless Communications and Networking Conference 2024. It is supported by the UK-funded project REASON and TUDOR under the Future Open Networks Research Challenge sponsored by DSIT.

**A Brief Description** 

The disaggregated and hierarchical architecture of advanced RAN presents significant challenges in efficiently placing baseband functions and user plane functions in conjunction with Multi-Access Edge Computing (MEC) to accommodate diverse 5G services. Therefore, this paper proposes a novel approach NetMind, which leverages Deep Reinforcement Learning (DRL) to determine the function placement strategies in RANs with diverse topologies, aiming at minimizing power consumption. 
NetMind formulates the function placement problem as a maze-solving task, enabling a Markov Decision Process with standardized action space scales across different networks. 
Additionally, a Graph Convolutional Network (GCN) based encoding mechanism is introduced, allowing features from different networks to be aggregated into a single RL agent. That facilitates the RL agent's generalization capability and minimizes the negative impact of retraining on power consumption.
In an example with three sub-networks, NetMind achieves comparable performance to traditional methods that require a dedicated DRL agent for each network, resulting in a 70% reduction in training costs. Furthermore, it demonstrates a substantial 32.76% improvement in power savings and a 41.67% increase in service stability compared to benchmarks from the existing literature.

**Handbook** 

For ease of reuse, the code is containerized and you can simply go inside the container by using the *docker* and *Dev Container* inserts.

Otherwise, you can also run the code based on related required dependents, listed in the *requirements.txt*

To use the code, there are two folders in src/SFC, namely *CODER* and *DRL*. *CODER* consists the the GCN-based encoder and decoder. 'src/SFC/CODER/EDcode/Network_x/data_generate_x.py' generates the data and 'src/SFC/CODER/EDcode/Network_x/meta_x.py'
train and save the encoders for reuse in the DRL training process.

Then *DRL* includes the agent and environment. It is realized based on the *gym*, and you can find the maze in 'src/SFC/DRL/maze/maze/envs/SfcEnv.py' and the main file in 'src/SFC/DRL/maze/main.py'

Refer to https://www.youtube.com/watch?v=kd4RrN-FTWY for setting up a gym-based environment. It's a good tutorial.


