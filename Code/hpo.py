from ConfigSpace import Configuration, ConfigurationSpace, Float
import numpy as np
from smac import HyperparameterOptimizationFacade, Scenario
from Crossvalidation import cross_validation
from pathlib import Path
import argparse
import pandas as pd
import os

"""
This file is the Hyper parameter optimization wrapper for this project.
given an argument ["bikelane", "one_shot", "2phase"] tunes the given task
"""
def test_config(config, seed: int = 0, save_model=False):
    """
    Function that executed the test given a configuration to be tested
    """
    if "quantile" in config.keys():
        quantile = config["quantile"]
    else: quantile = 0.9

    # if we tune the 2 phas model, we don't set the model, that was decided during bike lane tuning
    if not (bikephasepath ==""):
        bikelane_results = pd.read_csv(bikephasepath.joinpath("..").joinpath("inputs_and_results.csv"))
        model=bikelane_results["model"][0]
        transformation=bikelane_results["model"][0]

    if "model" in config.keys():
        model = config["model"]

    if "transformation" in config.keys():
        model = config["transformation"]

    payload = {"min_epochs": 100, "max_patience": 50, "oversampling_rate": config["oversampling_rate"],
               "resolution": 256, "transformation": transformation, "task": task,
               "model": model, "params": "full", "weights": 1,
               "optimizer": config["optimizer"], "lr": config["lr"], "stages": stages,
               "results_folder": results_directory,
               "logging": True, "quantile": quantile, "save_model": save_model,
               "bikephasepath": bikephasepath,
               "phase": "train"}

    # define the evaluation metric as return
    _, loss, _ = cross_validation(payload, seed)
    return loss


if __name__ == "__main__":
    # find the task to be tuned
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["bikelane", "one_shot", "2phase"], default="one_shot")
    args = parser.parse_args()

    output_directory = Path("../Results/smac_test")
    results_directory = Path("../Results/runs_test")
    output_directory = output_directory.joinpath(args.task)
    results_directory = results_directory.joinpath(args.task)

    if args.task == "bikelane":
        configspace = ConfigurationSpace({"transformation": ["rotations", "colorJitter", "gBlur", "all"],
                                          "model": ["resnet34", "resnet50", "resnet18"],
                                          "optimizer": ["RMSProp", "Adam", "SGD"]
                                          })
        lr = Float("lr", (0.00001, 0.01), log=True, default=0.001)
        configspace.add_hyperparameter(lr)

        oversampling_rate = Float("oversampling_rate", (0, 1), default=0.9)
        configspace.add_hyperparameter(oversampling_rate)

        task = "bikelane"
        stages = 1
        bikephasepath = ""

    elif args.task == "one_shot":

        configspace = ConfigurationSpace({"transformation": ["rotations", "colorJitter", "gBlur", "all"],
                                          "model": ["resnet34", "resnet50", "resnet18"],
                                          "optimizer": ["RMSProp", "Adam", "SGD"]
                                          })

        lr = Float("lr", (0.00001, 0.01), log=True, default=0.001)
        configspace.add_hyperparameter(lr)

        oversampling_rate = Float("oversampling_rate", (0, 1), default=0.9)
        configspace.add_hyperparameter(oversampling_rate)

        task = "one_shot"
        stages = 1
        bikephasepath = ""
    elif args.task == "2phase":

        configspace = ConfigurationSpace({"optimizer": ["RMSProp", "Adam", "SGD"]})
        lr = Float("lr", (0.00001, 0.01), log=True, default=0.001)
        quantile = Float("quantile", (0.3, 1), default=0.9)
        oversampling_rate = Float("oversampling_rate", (0,1), default=0.9)

        configspace.add_hyperparameter(oversampling_rate)
        configspace.add_hyperparameter(lr)
        configspace.add_hyperparameter(quantile)

        task = "one_shot"
        stages = 2

        bikephasepath = Path("../Results/").joinpath("bikelane" + "_tuned").joinpath("Config_1")

    # set function time limit to 5 days
    # time_limit = 5 * 24 * 60 * 60
    time_limit = 60

    scenario = Scenario(configspace, deterministic=True, n_trials=200, walltime_limit=time_limit,
                        output_directory=output_directory)
    smac = HyperparameterOptimizationFacade(scenario, test_config)
    incumbent = smac.optimize()


    # performing the best configuration experiment and saving the model files while at it.
    results_directory = Path("../Results/").joinpath(args.task + "_tuned")
    loss = test_config(incumbent, save_model=True)


    #the rest of this code is copied out from test_config to replicate and save the payload
    if "quantile" in incumbent.keys():
        quantile = incumbent["quantile"]
    else: quantile = 0.9

    # if we tune the 2 phase model, we don't set the model, that was decided during bike lane tuning
    if not (bikephasepath ==""):
        bikelane_results = pd.read_csv(bikephasepath.joinpath("..").joinpath("inputs_and_results.csv"))
        model=bikelane_results["model"][0]
        transformation=bikelane_results["model"][0]

    if "model" in incumbent.keys():
        model = incumbent["model"]

    if "transformation" in incumbent.keys():
        model = incumbent["transformation"]

    payload = {"min_epochs": 100, "max_patience": 50, "oversampling_rate": incumbent["oversampling_rate"],
               "resolution": 256, "transformation": transformation, "task": task,
               "model": model, "params": "full", "weights": 1,
               "optimizer": incumbent["optimizer"], "lr": incumbent["lr"], "stages": stages,
               "results_folder": results_directory,
               "logging": True, "quantile": quantile, "save_model": True,
               "bikephasepath": bikephasepath,
               "phase": "train"}
    result = pd.DataFrame(payload, index=[0])
    result.to_csv(os.path.join(results_directory, "inputs_and_results.csv"))

