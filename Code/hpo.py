from ConfigSpace import Configuration, ConfigurationSpace
import numpy as np
from smac import HyperparameterOptimizationFacade, Scenario
from Crossvalidation import cross_validation

def test_train(config, seed: int=0):
    payload = {"min_epochs": 70, "max_patience": 30, "oversampling_rate": config["oversampling_rate"],
               "resolution": 256, "transformation": config["transformation"], "task": "bikelane",
               "model": config["model"], "pretrained": True, "params": "full", "weights": config["weights"],
               "optimizer": config["optimizer"], "lr": config["lr"], "stages": 1, "results_folder": "../Results/smac3_runs",
               "logging": False}

    # define the evaluation metric as return
    _, loss, _ = cross_validation(payload, seed)
    return loss
configspace = ConfigurationSpace({"oversampling_rate":(1,100),
                                  "transformation": ["rotations", "colorJitter", "gBlur", "all"],
                                  "model": ["resnet34","resnet50"],
                                  "weights":(1,50),
                                  "optimizer": ["RMSProp","Adam"],
                                  "lr":(0.00001, 0.01)
                                 })
scenario = Scenario(configspace, deterministic=True, n_trials=100)
smac = HyperparameterOptimizationFacade(scenario, test_train)
incumbent = smac.optimize()