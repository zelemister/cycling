from ConfigSpace import Configuration, ConfigurationSpace, Float
import numpy as np
from smac import HyperparameterOptimizationFacade, Scenario
from Crossvalidation import cross_validation
from pathlib import Path

def test_train(config, seed: int=0):
    payload = {"min_epochs": 100, "max_patience": 50, "oversampling_rate": config["oversampling_rate"],
               "resolution": 256, "transformation": "all", "task": "one_shot",
               "model": "resnet34", "pretrained": True, "params": "full", "weights": 1,
               "optimizer": config["optimizer"], "lr": config["lr"], "stages": 1, "results_folder": "../Results/smac_runs/2Phase_Rims_2",
               "logging": True, "quantile":config["quantile"], "save_model":False}

    # define the evaluation metric as return
    _, loss, _ = cross_validation(payload, seed)
    return loss

configspace = ConfigurationSpace({"oversampling_rate":(1,1000),
                                  "optimizer":["RMSProp", "Adam", "SGD"]
                                  })

lr = Float("lr", (0.00001, 0.01), log=True, default=0.001)
quantile = Float("quantile", (0.3, 1), default= 0.9)
configspace.add_hyperparameter(lr)
configspace.add_hyperparameter(quantile)


scenario = Scenario(configspace, deterministic=True, n_trials=200, output_directory=Path("../Results/smac3_output/2_Phase_2"))
smac = HyperparameterOptimizationFacade(scenario, test_train)
incumbent = smac.optimize()