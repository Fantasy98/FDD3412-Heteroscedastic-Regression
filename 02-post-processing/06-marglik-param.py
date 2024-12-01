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
filename = "input-uci-marglik-study.csv"
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
            'natural_eb',
            # 'naive_eb',        
            'natural_gs',        
        ]

configs = model_configs['natural_gs_pp']
#----------------------------
# HYPER-PARAM
#----------------------------
param_data = data
# HYPER-PARAM
#----------------------------
marglik_frequencys =[25,50,100,150]             # Default=50
n_hyperstepses  = [25,50,100,150]               # Default=50
approxs = ['full','kron','diag','kernel']   # Default=full
#----------------------------

# Plot heatmap again after removing large values
fig, axs= plt.subplots(1,2,figsize=(14, 6),sharex=True,sharey=True)

#---------------------------------------------------------------------
for il,case in enumerate(CASES):
    configs = model_configs[case]

    param_data_ = param_data[
                            (param_data['activation'] == 'gelu') &\
                            (param_data['likelihood'] == configs['likelihood']) &\
                            (param_data['head'] == configs['head']) &\
                            (param_data['method'] == configs['method']) &\
                            (param_data['approx'] == 'full')
                            ]

    # Re-pivot data for heatmap analysis
    heatmap_data_cleaned = param_data_.pivot_table(
        index='marglik_frequency', 
        columns='n_hypersteps', 
        values=configs['result'], 
        aggfunc='mean')

    sns.heatmap(heatmap_data_cleaned, 
                annot=True, 
                cmap='Reds', 
                fmt=".3f", 
                cbar=False,
                ax=axs[il],
                # cbar_kws={'label': 'LL'},
                )

    axs[il].set(
        **{
        'xlabel':'Marglik Freq',
        'ylabel':'Hyper Steps',
        'title':configs['label'],
        }
        )

fig.savefig('figs/06-param-study-Marglik-Freq-Hypersteps.jpg',bbox_inches='tight',dpi=300)
