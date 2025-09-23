import wandb
import pandas as pd

project = "siqi_verl_gamellm"                   # 项目名称
id = "0910_math_cobb"     # run_id
api = wandb.TrackingApi()        
run = api.run(project=project, run_id=id)