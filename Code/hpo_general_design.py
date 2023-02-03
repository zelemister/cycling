from ConfigSpace import Configuration, ConfigurationSpace, Float
import numpy as np
from smac import HyperparameterOptimizationFacade, Scenario
from Crossvalidation import cross_validation
from pathlib import Path
def test_train(config, seed: int=0):
    payload = {"min_epochs": 70, "max_patience": 30, "oversampling_rate": config["oversampling_rate"],
               "resolution": 256, "transformation": config["transformation"], "task": "one_shot",
               "model": config["model"], "pretrained": True, "params": "full", "weights": 1,
               "optimizer": "Adam", "lr": config["lr"], "stages": 1, "results_folder": "../Results/smac_runs/Rims",
               "logging": True}

    # define the evaluation metric as return
    _, loss, _ = cross_validation(payload, seed)
    return loss
configspace = ConfigurationSpace({"oversampling_rate":(1,100),
                                  "transformation": ["rotations", "colorJitter", "gBlur", "all"],
                                  "model": ["resnet34","resnet50", "resnet18"],
                                  "optimizer":["RMSProp", "Adam", "SGD"]
                                  })

lr = Float("lr", (0.00001, 0.01), log=True, default=0.001)
configspace.add_hyperparameter(lr)

scenario = Scenario(configspace, deterministic=True, n_trials=200, output_directory=Path("../Results/smac3_output/One_shot"))
smac = HyperparameterOptimizationFacade(scenario, test_train)
incumbent = smac.optimize()