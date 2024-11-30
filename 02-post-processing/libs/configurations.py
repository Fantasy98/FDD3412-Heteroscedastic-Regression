"""
Configuration of training model 
@yuningw
"""

UCI = ['boston-housing', 'concrete', 'energy', 'kin8nm', 'naval-propulsion-plant', 'power-plant', 'wine-quality-red', 'yacht']
CRISPR = ['flow-cytometry-HEK293','survival-screen-A375','survival-screen-HEK293']
IMAGE_REG = ['mnist','fmnist','cifar10']

model_configs={
            'homo':{
                    'likelihood':"homoscedastic",
                    'head':'natural',
                    'method':"marglik",
                    'name':'Homoscedastic',
                    'label':'Homoscedastic | GS | PP',
                    'regularization':'EB',
                    'if_posterior':'Y',
                    'result':'test/loglik',
                  },
            
            'naive_gs':{
                    'likelihood':"heteroscedastic",
                    'head':"meanvar",
                    'method':"map",
                    'name':'Naive NLL',
                    'label':'Navie NLL',
                    'regularization':'GS',
                    'if_posterior':'N',
                    'result':'test/loglik',
                  },
            
            'betahalf':{
                    'likelihood':"heteroscedastic",
                    'head':"gaussian",
                    'method':"betanll",
                    'beta':0.5,
                    'name': r"$\beta$-"+'NLL (0.5)',
                    'label':r"$\beta$-"+'NLL (0.5)',
                    'regularization':'GS',
                    'if_posterior':'N',
                    'result':'test/loglik',
                  },
            'betaone':{
                  'likelihood':"heteroscedastic",
                    'head':"gaussian",
                    'method':"betanll",
                    'beta':1,
                    'name':r"$\beta$-"+'NLL (1.0)',
                    'label':r"$\beta$-"+'NLL (1.0)',
                    'regularization':'GS',
                    'if_posterior':'N',
                    'result':'test/loglik',
                  },

            'faith':{
                  'likelihood':"heteroscedastic",
                    'head':"gaussian",
                    'method':"faithful",
                    'name':'Faithful',
                    'label':'Faithful',
                    'regularization':'GS',
                    'if_posterior':'N',
                    'result':'test/loglik',
                  },

            'mcdropout':{
                    'likelihood':"heteroscedastic",
                    'head':"gaussian",
                    'method':"mcdropout",
                    'name':'MC-Dropout',
                    'label':'MC-Dropout',
                    'regularization':'GS',
                    'if_posterior':'N',
                    'result':'test/loglik',
                  },
            
            'vi':{
                    'likelihood':"heteroscedastic",
                    'head':"gaussian",
                    'method':"vi",
                    'name':'VI',
                    'label':'VI | PP',
                    'regularization':'GS',
                    'if_posterior':'N',
                    'result':'test/loglik',
                  },
            
            'naive_eb_pp':{
                    'likelihood':"heteroscedastic",
                    'head':"meanvar",
                    'method':"marglik",
                    'name':'Naive NLL',
                    'label':'Naive NLL | EB | PP',
                    'regularization':'EB',
                    'if_posterior':'Y',
                    'result':'test/loglik_bayes',
                  },
            
            'naive_eb':{
                    'likelihood':"heteroscedastic",
                    'head':"meanvar",
                    'method':"marglik",
                    'name':'Naive NLL',
                    'label':'Naive NLL | EB ',
                    'regularization':'EB',
                    'if_posterior':'N',
                    'result':'test/loglik',
                  },

            'natural_gs_pp':{
                    'likelihood':"heteroscedastic",
                    'head':"natural",
                    'method':"map",
                    'name':'Natural NLL',
                    'label':'Natural NLL | GS | PP',
                    'regularization':'GS',
                    'if_posterior':'Y',
                    'result':'test/loglik_bayes',
                  },
  
            'natural_gs':{
                    'likelihood':"heteroscedastic",
                    'head':"natural",
                    'method':"map",
                    'name':'Natural NLL',
                    'label':'Natural NLL | GS',
                    'regularization':'GS',
                    'if_posterior':'N',
                    'result':'test/loglik',
                  },

            
            'natural_eb_pp':{
                    'likelihood':"heteroscedastic",
                    'head':"natural",
                    'method':"marglik",
                    'name':'Natural NLL',
                    'label':'Natural NLL | EB | PP',
                    'regularization':'EB',
                    'if_posterior':'Y',
                    'result':'test/loglik_bayes',
                  },
  
            'natural_eb':{
                    'likelihood':"heteroscedastic",
                    'head':"natural",
                    'method':"marglik",
                    'name':'Natural NLL',
                    'label':'Natural NLL | EB ',
                    'regularization':'EB',
                    'if_posterior':'N',
                    'result':'test/loglik',
                  },

            
            }