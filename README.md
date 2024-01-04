# NetMind

This work has been accepted by IEEE Wireless Communications and Networking Conference 2024.

For ease of reuse, the code is containerized and you can simply go inside the container by using the *docker* and *Dev Container* insert.

Related required dependents are listed in the *requirements.txt*

To use the code, there are two folders in src/SFC, namely *CODER* and *DRL*. *CODER* consists the the GCN-based encoder and decoder, they save the model for reuse in the DRL training process.

Then *DRL* includes the agent and environment. It is realized based on the *gym*, and you can find the maze in 'src/SFC/DRL/maze/maze/envs/SfcEnv.py'

You can refer to this https://www.youtube.com/watch?v=kd4RrN-FTWY for setting up a gym-based environment. It's a good tutorial.


