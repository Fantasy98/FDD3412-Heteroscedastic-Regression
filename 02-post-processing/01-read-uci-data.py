"""
Notebook for post-processing the results
"""
import os
import numpy as np
import pandas as pd
from libs.configurations import * 
database = "./database/"
filename = "input-uci-crispr-experiments.csv"

# Read Data
data = pd.read_csv(os.path.join(database,filename))
## Sort out the job that has been finished
finished_data = data[data['State'] == 'finished']
print(f"[IO] JOB {len(finished_data)}/{len(data)} SORTED")

# Initialize DataFrame
table_updated = pd.DataFrame(columns=["Objective", "Regularization", "Posterior Predictive"] + UCI)

# Traversing all the database
for jl, dataset in enumerate(UCI):
  data_class = finished_data[(finished_data['dataset'] == dataset)]
  for il, nn in enumerate(model_configs.keys()):
    data = data_class[(data_class['method']==model_configs[nn]['method']) &\
                      (data_class['head']==model_configs[nn]['head']) &\
                      (data_class['likelihood']==model_configs[nn]['likelihood'])
                      ]
    if 'beta' in nn: 
      data = data[data['beta']==model_configs[nn]['beta']]
    result_arrary=data[model_configs[nn]['result']].to_numpy() 
    # get rid of the abnormal data 
    result_arrary = result_arrary[np.where(result_arrary >=-5.0)]
    result_arrary_mean = result_arrary.mean()
    result_arrary_std  = result_arrary.std()
    print(f'[IO] Dataset:{dataset}, Method ={nn}, NLL={result_arrary_mean} std = {result_arrary_std}')

    table_updated.loc[il,'Objective']= model_configs[nn]['name']
    table_updated.loc[il,'Regularization']= model_configs[nn]['regularization']
    table_updated.loc[il,'Posterior Predictive']= model_configs[nn]['if_posterior']
    table_updated.loc[il,dataset]=f"{result_arrary_mean:.2f} ({result_arrary_std:.2f})"

table_updated.to_csv("outputs/output-01-uci.csv")
latex_table = table_updated.to_latex(index=False,  # Exclude index
    escape=False,  # Allow LaTeX commands
    column_format="|l|l|l|c|c|c|c|c|c|c|c|c|c|",  # NeurIPS style
    caption="Performance of Methods on Various Datasets",  # Add a caption
    label="tab:results",  # Add a label for referencing
    )
with open("outputs/output_uci_table.tex", "w") as f:
    f.write(latex_table)
f.close()