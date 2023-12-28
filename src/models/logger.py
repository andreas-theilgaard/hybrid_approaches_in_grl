from omegaconf import OmegaConf
import pandas as pd
from src.models.metrics import directions
import wandb
import time

class WB_Logger:
    def __init__(self,config):
        self.config = config

    def init_WB(self,seed):
        wandb.init(
            project="Hybrid Approaches in GRL",
            reinit=False,
            group=f"{self.config.dataset.task}-{self.config.dataset.dataset_name}-{self.config.model_type}-{self.config.dataset.DIM}",
            job_type=f"{seed}",
            anonymous='must',
            save_code=True,
            settings=wandb.Settings(start_method="thread"),
            config=OmegaConf.to_container(cfg=self.config,resolve=True,throw_on_missing=True),
            tags=[self.config.dataset.task,self.config.dataset.dataset_name,self.config.model_type,str(self.config.dataset.DIM)]
        )

    def log_metrics(metrics):
        for x in metrics:
            if x != 'Run':
                data_type, metric = x.lower().split(" ")
                wandb.log({f"{data_type} {metric}:": metrics[x]})
    
    def log_embedding():
        pass

class LoggerClass(object):
    def __init__(self, runs=None, metrics=None, track_metric=None, seeds=None, log=None, track_best=False,config=None):
        self.results = pd.DataFrame()
        self.metrics = metrics
        self.track_metric = track_metric
        self.track_best = track_best
        self.runs = runs
        self.current_run = 0
        self.saved_results = None
        self.seeds = seeds
        self.log = log
        self.saved_values = {}
        self.directions = [directions(x) for x in metrics]
        self.config = config
        self.WB_Logger = WB_Logger(config=self.config)

        if self.track_metric and self.track_best:
            self.best_val = -1

    def start_run(self):
        if self.config.use_wandb and not self.config.debug:
            self.WB_Logger.init_WB(seed=self.seeds[self.current_run-1])
        self.current_run += 1
        self.start = time.time()
        self.log.info(f"Run {self.current_run}/{self.runs} using seed {self.seeds[self.current_run-1]}")
        self.X = pd.DataFrame(columns=self.metrics + ["Run"])

    def add_to_run(self, loss: float, results: dict):
        results_to_add = {}
        results_to_add["Run"] = self.current_run
        for x in self.metrics:
            data_type, metric = x.lower().split(" ")
            if metric == "loss":
                results_to_add[x] = loss
            else:
                results_to_add[x] = results[data_type][metric]
        self.X.loc[len(self.X), :] = results_to_add

        if self.config.use_wandb and not self.config.debug:
            WB_Logger.log_metrics(metrics=results_to_add)

        if self.track_best and self.track_metric:
            if results["val"][self.track_metric] > self.best_val:
                self.best_val = results["val"][self.track_metric]
                return True
            return False

    def end_run(self):
        end =  time.time()
        self.results = pd.concat([self.results, pd.DataFrame(self.X)])
        elapsed = end - self.start
        if self.config.use_wandb and not self.config.debug:
            wandb.log({f"Elapsed Time:": elapsed})

    def save_value(self, values: dict):
        for key in values.keys():
            self.saved_values[key] = values[key]

    def save_results(self, save_path):
        self.results.columns = self.metrics + ["Run"]
        self.results.reset_index(drop=True, inplace=True)
        self.results.to_json(save_path)
        if self.config.use_wandb and not self.config.debug:
            results_table = wandb.Table(dataframe=self.results)
            wandb.log({"Results": results_table})
            wandb.finish()


    def logger_load(self, save_path):
        self.saved_results = pd.read_json(save_path)

    def get_statistics(self, metrics: list, out=False):
        assert sum([1 for x in self.directions if x in ["+", "-"]]) == len(
            self.directions
        ), "direction should be either ['+','-'] depending on the metric should be maximized or minimized"
        results = self.results.copy()
        if isinstance(self.saved_results, pd.DataFrame):
            results = self.saved_results.copy()

        results_for_metrics = {}
        for metric, direction in zip(metrics, self.directions):
            if direction == "+":
                best_results = results.groupby("Run")[metric].max()
                if "Test" in metric:
                    try:
                        best_val_runs = dict(results.groupby("Run").max()["Val " + (metric.split(" "))[1]])
                        best_test_result = results[
                            (results["Run"].isin(list(best_val_runs.keys())))
                            & (results["Val " + (metric.split(" "))[1]].isin(list(best_val_runs.values())))
                        ][metric]
                    except:
                        print("Error encountered")
                        best_test_result = best_results.copy()
            elif direction == "-":
                best_results = results.groupby("Run")[metric].min()
                if "Test" in metric:
                    try:
                        best_val_runs = dict(results.groupby("Run").min()["Val " + (metric.split(" "))[1]])
                        best_test_result = results[
                            (results["Run"].isin(list(best_val_runs.keys())))
                            & (results["Val " + (metric.split(" "))[1]].isin(list(best_val_runs.values())))
                        ][metric]
                    except:
                        print("Error encountered")
                        best_test_result = best_results.copy()
            else:
                raise ValueError("Direction should be either '+' or '-'")

            last_results = results.groupby("Run")[metric].tail(1)
            arrow = "↓" if direction == "-" else "↑"
            print("")
            if "Test" in metric:
                self.log.info(
                    f"""\nMetric: {metric} {arrow}:\n   Best Result: {best_results.mean():.4f} ± {best_results.std():.4f}\n   Last Result: {last_results.mean():.4f} ± {last_results.std():.4f}\n   ----------------------------\n   Best Test Result: {best_test_result.mean():.4f} ± {best_test_result.std():.4f}"""
                )

            else:
                self.log.info(
                    f"""\nMetric: {metric} {arrow}:\n   Best Result: {best_results.mean():.4f} ± {best_results.std():.4f}\n   Last Result: {last_results.mean():.4f} ± {last_results.std():.4f}"""
                )
            print("")
            results_for_metrics[metric] = best_results

        if out:
            return results_for_metrics
        