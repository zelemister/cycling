from ConfigSpace import Configuration, ConfigurationSpace, Float
import numpy as np
from smac import HyperparameterOptimizationFacade, Scenario
from Crossvalidation import cross_validation
from pathlib import Path
import argparse

def test_config(config, seed: int=0, save_model=False):
    payload = {"min_epochs": 100, "max_patience": 50, "oversampling_rate": config["oversampling_rate"],
               "resolution": 256, "transformation": config["transformation"], "task": task,
               "model": config["model"], "params": "full", "weights": 1,
               "optimizer": config["optimizer"], "lr": config["lr"], "stages": stages, "results_folder": results_directory,
               "logging": True, "quantile": config["quantile"], "save_model": save_model, "bikephasepath":bikephasepath,
               "phase":"train"}

    # define the evaluation metric as return
    #_, loss, _ = cross_validation(payload, seed)
    loss = 0.2
    return loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["bikelane", "one_shot", "2phase"], default="one_shot")
    args = parser.parse_args()

    output_directory = Path("../Results/smac_test")
    results_directory = Path("../Results/runs_test")
    if args.task == "bikelane":
        output_directory = output_directory.joinpath("bikelane")
        results_directory = results_directory.joinpath("bikelane")
        configspace = ConfigurationSpace({"transformation": ["rotations", "colorJitter", "gBlur", "all"],
                                  "model": ["resnet34","resnet50", "resnet18"],
                                  "optimizer":["RMSProp", "Adam", "SGD"]
                                  })
        lr = Float("lr", (0.00001, 0.01), log=True, default=0.001)
        configspace.add_hyperparameter(lr)

        oversampling_rate = Float("oversampling_rate", (0,1), default=0.1)
        configspace.add_hyperparameter(oversampling_rate)

        task = "bikelane"
        stages = 1
        bikephasepath = ""
    elif args.task == "one_shot":
        output_directory = output_directory.joinpath("one_shot")
        results_directory = results_directory.joinpath("one_shot")


        configspace = ConfigurationSpace({"transformation": ["rotations", "colorJitter", "gBlur", "all"],
                                          "model": ["resnet34", "resnet50", "resnet18"],
                                          "optimizer":["RMSProp", "Adam", "SGD"]
                                          })

        lr = Float("lr", (0.00001, 0.01), log=True, default=0.001)
        configspace.add_hyperparameter(lr)

        oversampling_rate = Float("oversampling_rate", (0,1), default=0.1)
        configspace.add_hyperparameter(oversampling_rate)

        task = "one_shot"
        stages = 1
        bikephasepath = ""
    elif args.task == "2phase":

        output_directory = output_directory.joinpath("2_Phase")
        results_directory = results_directory.joinpath("2_Phase")


        configspace = ConfigurationSpace({"optimizer":["RMSProp", "Adam", "SGD"],
                                          "model":"resnet34",
                                          "transformation":["rotations"]
                                          })
        lr = Float("lr", (0.00001, 0.01), log=True, default=0.001)
        quantile = Float("quantile", (0.3, 1), default= 0.9)

        oversampling_rate = Float("oversampling_rate", (0,1), default=0.1)
        configspace.add_hyperparameter(oversampling_rate)

        configspace.add_hyperparameter(lr)
        configspace.add_hyperparameter(quantile)

        task = "one_shot"
        stages = 2

        #this has to be set
        bikephasepath = ""

        #the model and transformation have to be set, how to constants work in smac?
        #change the payload
    #set function time limit to 5 days
    #time_limit = 5 * 24 * 60 * 60
    time_limit = 60 * 60

    scenario = Scenario(configspace, deterministic=True, n_trials=200, walltime_limit = time_limit,
                        output_directory=output_directory)
    smac = HyperparameterOptimizationFacade(scenario, test_config)
    incumbent = smac.optimize()

    best_model = Path("../Results/").joinpath(args.task + "_tuned")
    _ = test_config(incumbent, save_model=True)
