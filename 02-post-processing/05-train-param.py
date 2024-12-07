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
filename = "input-uci-train-param.csv"
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
            'naive_eb_pp',        
            # 'natural_gs_pp',        
        ]

configs = model_configs['natural_gs_pp']
#----------------------------
# HYPER-PARAM
#----------------------------
param_data = data

# Plot heatmap again after removing large values
fig, axs= plt.subplots(1,2,figsize=(14, 6),sharex=True,sharey=True)

#---------------------------------------------------------------------
for il,case in enumerate(CASES):
    configs = model_configs[case]

    param_data_ = param_data[(param_data['test/loglik'] > -10) &\
                            (param_data['activation'] == 'gelu') &\
                            (param_data['likelihood'] == configs['likelihood']) &\
                            (param_data['head'] == configs['head']) &\
                            (param_data['method'] == configs['method']) &\
                            (param_data['lr_min'] == 1e-5) &\
                            (param_data['lr_hyp'] == 1e-2) &\
                            (param_data['lr_hyp_min'] == 1e-3)
                            ]

    # Re-pivot data for heatmap analysis
    heatmap_data_cleaned = param_data_.pivot_table(
        index='lr', 
        columns='batch_size', 
        values=configs['result'], 
        aggfunc='mean')

    sns.heatmap(heatmap_data_cleaned, 
                annot=True, 
                cmap='Greens', 
                fmt=".2f", 
                cbar=False,
                ax=axs[il],
                # cbar_kws={'label': 'LL'},
                )

    axs[il].set(
        **{
        'xlabel':'batch size',
        'ylabel':'max lr',
        'title':configs['label'],
        }
        )

fig.savefig('figs/05-param-study-lr-batch.jpg',bbox_inches='tight',dpi=300)

# #---------------------------------------------------------------------
# fig, axs= plt.subplots(1,2,figsize=(14, 6),sharex=True,sharey=True)
# for il,case in enumerate(CASES):
#     configs = model_configs[case]

#     param_data_ = param_data[(param_data['test/loglik'] > -10) &\
#                             (param_data['activation'] == 'gelu') &\
#                             (param_data['likelihood'] == configs['likelihood']) &\
#                             (param_data['head'] == configs['head']) &\
#                             (param_data['method'] == configs['method']) &\
#                             (param_data['lr_hyp_min'] == 1e-3) &\
#                             (param_data['lr_hyp'] == 1e-2)
#                             ]

#     # Re-pivot data for heatmap analysis
#     heatmap_data_cleaned = param_data_.pivot_table(
#         index='lr_min', 
#         columns='lr', 
#         values=configs['result'], 
#         aggfunc='mean')

#     sns.heatmap(heatmap_data_cleaned, 
#                 annot=True, 
#                 cmap='Greens', 
#                 fmt=".2f", 
#                 cbar=False,
#                 ax=axs[il],
#                 # cbar_kws={'label': 'LL'},
#                 )

#     axs[il].set(
#         **{
#         'xlabel':'max lr',
#         'ylabel':'min lr',
#         'title':configs['label'],
#         }
#         )

# fig.savefig('figs/05-param-study-lr.jpg',bbox_inches='tight',dpi=300)

# fig, axs= plt.subplots(1,2,figsize=(14, 6),sharex=True,sharey=True)
# #---------------------------------------------------------------------
# for il,case in enumerate(CASES):
#     configs = model_configs[case]

#     param_data_ = param_data[(param_data['test/loglik'] > -10) &\
#                             (param_data['activation'] == 'gelu') &\
#                             (param_data['likelihood'] == configs['likelihood']) &\
#                             (param_data['head'] == configs['head']) &\
#                             (param_data['method'] == configs['method']) &\
#                             (param_data['lr_min'] == 1e-5) &\
#                             (param_data['lr'] == 1e-3) &\
#                             (param_data['batch_size'] == 256) 
#                             ]

#     # Re-pivot data for heatmap analysis
#     heatmap_data_cleaned = param_data_.pivot_table(
#         index='lr_hyp_min', 
#         columns='lr_hyp', 
#         values=configs['result'], 
#         aggfunc='mean')

#     sns.heatmap(heatmap_data_cleaned, 
#                 annot=True, 
#                 cmap='Greens', 
#                 fmt=".2f", 
#                 cbar=False,
#                 ax=axs[il],
#                 # cbar_kws={'label': 'LL'},
#                 )

#     axs[il].set(
#         **{
#         'xlabel':'max hyper lr',
#         'ylabel':'min hyper lr',
#         'title':configs['label'],
#         }
#         )

# fig.savefig('figs/05-param-study-hyp_lr.jpg',bbox_inches='tight',dpi=300)

