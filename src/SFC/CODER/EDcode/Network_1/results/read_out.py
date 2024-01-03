import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import os
from scipy import stats

matplotlib.rcParams['font.family'] = 'TeX Gyre Pagella'

current_dir = os.path.abspath(os.path.dirname(__file__))

chkpt_dir = current_dir + "/data_loss"

check_data_reward = pd.read_csv(chkpt_dir)
# '/Users/ocean/Library/Mobile Documents/com~apple~CloudDocs/Documents/CUDU/Code_1/Multi-results/Results/data_reward')
r = check_data_reward["loss"]

plt.figure(figsize=(6.5, 5))

plt.grid(linestyle="-.")
plt.xticks(size=15)
plt.yticks(size=15)
plt.xlabel("Epoch", fontsize=15)
plt.ylabel("MSE loss", fontsize=15)

plt.plot(r, color="cornflowerblue")

plt.savefig(current_dir + "/training.png", dpi=300, bbox_inches="tight")

plt.show()
