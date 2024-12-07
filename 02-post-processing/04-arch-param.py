"""
Notebook for post-processing the results
"""
import os
import numpy as np
import pandas as pd
from libs.configurations import * 
from libs.plot import * 
import matplotlib.pyplot as plt 
import seaborn as sns
plt_setUp()
colors = plt.cm.tab20.colors

database = "./database/"
filename = "input-uci-arch-param.csv"
output_name ='03-image'
# Read Data
data = pd.read_csv(os.path.join(database,filename))
## Sort out the job that has been finished
data = data[data['State'] == 'finished']
print(f"[IO] JOB {len(data)}/{len(data)} SORTED")


#----------------------------
# HYPER-PARAM
#----------------------------
CASES = [
            'natural_eb_pp',
            'naive_eb_pp'        
        ]

configs = model_configs['natural_gs_pp']
#----------------------------
# HYPER-PARAM
#----------------------------
widths=[25,50,75]   # Origin=50
depths=[1,2,3]      # Origin=1
activations=['gelu','selu','silu'] # Origin gelu
columns_of_interest = ['test/loglik', 'width', 'depth', 'activation']
param_data = data

# Plot heatmap again after removing large values
fig, axs= plt.subplots(1,2,figsize=(14, 6),sharex=True,sharey=True)

#---------------------------------------------------------------------
for il,case in enumerate(CASES):
    configs = model_configs[case]

    param_data_ = param_data[(param_data['test/loglik_bayes'] > -2) &\
                            (param_data['activation'] == 'gelu') &\
                            (param_data['likelihood'] == configs['likelihood']) &\
                            (param_data['head'] == configs['head']) &\
                            (param_data['method'] == configs['method'])
                            ]

    # Re-pivot data for heatmap analysis
    heatmap_data_cleaned = param_data_.pivot_table(
        index='depth', 
        columns='width', 
        values=configs['result'], 
        aggfunc='mean')

    sns.heatmap(heatmap_data_cleaned, 
                annot=True, 
                cmap='Blues', 
                fmt=".2f", 
                cbar=False,
                ax=axs[il],
                # cbar_kws={'label': 'LL'},
                )

    axs[il].set_title(configs['label'])
fig.savefig('figs/04-param-study-width-depth.jpg',bbox_inches='tight',dpi=300)

