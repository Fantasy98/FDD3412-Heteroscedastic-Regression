"""
Notebook for post-processing the results
"""
import os
import numpy as np
import pandas as pd

database = "./database/"
filename = "input-uci-crispr-experiments.csv"

# Read Data
data = pd.read_csv(os.path.join(database,filename))
## Sort out the job that has been finished
finished_data = data[data['State'] == 'finished']

def get_mean_std(dt):
    """Calculate the Mean and Std of the results"""
    s = []
    for col in dt.columns:
        s.append((np.mean(dt[col].to_numpy()), np.std(dt[col].to_numpy())))
    return s


UCI = ['boston-housing', 'concrete', 'energy', 'kin8nm', 'naval-propulsion-plant', 'power-plant', 'wine-quality-red', 'yacht']
ls = []

for dataset in UCI:
    result = {}
    data_class = finished_data[(finished_data['dataset'] == dataset)]

    # homoscedastic
    homo_data = data_class[(data_class['likelihood'] == 'homoscedastic') & (data_class['method'] == 'marglik')]
    homo_pp = np.array(homo_data['test/loglik_bayes'])
    result['homo'] = get_mean_std(homo_pp)

    naive_data = data_class[(data_class['likelihood'] == 'heteroscedastic') &\
                           (data_class['head'] == 'meanvar') & (data_class['method'] == 'map')]
    naive_lik = np.array(naive_data['test/loglik'])
    result['naive_gs'] = get_mean_std(naive_lik)

    # Other methods
    betahalf_data = data_class[(data_class['likelihood'] == 'heteroscedastic') &\
                                (data_class['method'] == 'betanll') & (data_class['beta'] == 0.5)]
    betahalf_lik = np.array(betahalf_data['test/loglik'])
    result['betahalf'] = get_mean_std(betahalf_lik)

    betaone_data = data_class[(data_class['likelihood'] == 'heteroscedastic') &\
                              (data_class['method'] == 'betanll') & (data_class['beta'] == 1)]
    betaone_lik = np.array(betaone_data['test/loglik'])
    result['betaone'] = get_mean_std(betaone_lik)

    faith_data = data_class[(data_class['likelihood'] == 'heteroscedastic') &\
                            (data_class['method'] == 'faithful')]
    faith_lik = np.array(faith_data['test/loglik'])
    result['faith'] = get_mean_std(faith_lik)
    
    mcdropout_data = data_class[(data_class['likelihood'] == 'heteroscedastic') &\
                                 (data_class['method'] == 'mcdropout')]
    mcdropout_lik = np.array(mcdropout_data['test/loglik'])
    result['mcdropout'] = get_mean_std(mcdropout_lik)

    vi_data = data_class[(data_class['likelihood'] == 'heteroscedastic') &\
                          (data_class['method'] == 'vi')]
    vi_lik = np.array(vi_data['test/loglik'])
    result['vi'] = get_mean_std(vi_lik)

    # naive
    naive_eb_pp_data = data_class[(data_class['likelihood'] == 'heteroscedastic') &\
                                   (data_class['head'] == 'meanvar') & (data_class['method'] == 'marglik')]
    naive_eb_pp_lik = np.array(naive_eb_pp_data['test/loglik_bayes'])
    result['naive_eb_pp'] = get_mean_std(naive_eb_pp_lik)

    naive_eb_data = data_class[(data_class['likelihood'] == 'heteroscedastic') &\
                                (data_class['head'] == 'meanvar') & (data_class['method'] == 'marglik')]
    naive_eb_lik = np.array(naive_eb_data['test/loglik'])
    result['naive_eb'] = get_mean_std(naive_eb_lik)

    # natural
    natural_gs_pp_data = data_class[(data_class['likelihood'] == 'heteroscedastic') &\
                                     (data_class['head'] == 'natural') & (data_class['method'] == 'map')]
    natural_gs_pp_lik = np.array(natural_gs_pp_data['test/loglik_bayes'])
    result['natural_gs_pp'] = get_mean_std(natural_gs_pp_lik)

    natural_gs_data = data_class[(data_class['likelihood'] == 'heteroscedastic') &\
                                  (data_class['head'] == 'natural') & (data_class['method'] == 'map')]
    natural_gs_lik = np.array(natural_gs_data['test/loglik'])
    result['natural_gs'] = get_mean_std(natural_gs_lik)

    natural_eb_pp_data = data_class[(data_class['likelihood'] == 'heteroscedastic') &\
                                     (data_class['head'] == 'natural') & (data_class['method'] == 'marglik')]
    natural_eb_pp_lik = np.array(natural_eb_pp_data['test/loglik_bayes'])
    result['natural_eb_pp'] = get_mean_std(natural_eb_pp_lik)

    natural_eb_data = data_class[(data_class['likelihood'] == 'heteroscedastic') &\
                                  (data_class['head'] == 'natural') & (data_class['method'] == 'marglik')]
    natural_eb_lik = np.array(natural_eb_data['test/loglik'])
    result['natural_eb'] = get_mean_std(natural_eb_lik)

    ls.append(result)


# Define the table columns
objective = ["Homoscedastic", 
             "Naive NLL", 
             "β-NLL (0.5)", 
             "β-NLL (1.0)", 
             "Faithful", 
             "MC-Dropout", 
             "VI", 
             "Naive NLL", 
             "Naive NLL", 
             "Natural NLL", 
             "Natural NLL", 
             "Natural NLL", 
             "Natural NLL"]
regularization = ["EB", 
                  "GS", 
                  "GS", 
                  "GS",
                  "GS",
                  "GS",
                  "GS",
                  "EB", 
                  "EB", 
                  "GS",
                  "GS",
                  "EB",
                  "EB"]
posterior_predictive = [
                  "Y",
                  "N",
                  "N",
                  "N",
                  "N",
                  "N",
                  "N",
                  "Y",
                  "N",
                  "Y",
                  "N",
                  "Y",
                  "N"]

# Initialize DataFrame
table_updated = pd.DataFrame(columns=["Objective", "Regularization", "Posterior Predictive"] + UCI)
table_updated["Objective"] = objective
table_updated["Regularization"] = regularization
table_updated["Posterior Predictive"] = posterior_predictive

# Populate the table with values from `ls`
model_configs = ["homo", "naive_gs", "betahalf", "betaone", "faith", 
                 "mcdropout", "vi", "naive_eb_pp", "naive_eb", 
                 "natural_gs_pp", "natural_gs", "natural_eb_pp", "natural_eb"]

for i, config in enumerate(model_configs):
    for j, dataset in enumerate(UCI):
        mean, std = ls[j].get(config, (np.nan, np.nan))
        if not np.isnan(mean) and not np.isnan(std):
            table_updated.loc[i, dataset] = f"{mean:.2f} ({std:.2f})"
        else:
            table_updated.loc[i, dataset] = "N/A"

# Print the table
print(table_updated.to_string(index=False))

table_updated.to_csv("results_table.csv")


