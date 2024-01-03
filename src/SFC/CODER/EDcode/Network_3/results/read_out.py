import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from scipy import stats
import matplotlib

matplotlib.rcParams['font.family'] = 'TeX Gyre Pagella'

current_dir = os.path.abspath(os.path.dirname(__file__))

chkpt_dir = current_dir + "/data_loss"

check_data_reward = pd.read_csv(chkpt_dir)
# '/Users/ocean/Library/Mobile Documents/com~apple~CloudDocs/Documents/CUDU/Code_1/Multi-results/Results/data_reward')
r = check_data_reward["loss"]

plt.figure(figsize=(6.5, 5))

plt.grid(linestyle="-.")
plt.xticks(size=14)
plt.yticks(size=14)
plt.xlabel("Batch", fontsize=14)
plt.ylabel("Loss", fontsize=14)

plt.plot(r,color='orange')

plt.savefig(current_dir + "/training_4.png", dpi=300, bbox_inches="tight")

plt.show()
