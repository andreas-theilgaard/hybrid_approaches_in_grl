from ogb.linkproppred import Evaluator as eval_link
from ogb.nodeproppred import Evaluator as eval_class
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    auc,
    roc_curve,
)
import torch
import copy

def directions(metric):
    if metric == "loss":
        return "-"
    return "+"


class METRICS:
    def __init__(self, metrics_list: list, task: str, dataset):
        self.metrics_list = metrics_list
        self.task = task
        if task == "LinkPrediction":
            if dataset in ["ogbl-collab", "ogbl-vessel", "ogbl-citation2"]:
                self.evaluator_link = eval_link(name=dataset)
            else:
                self.evaluator_link = eval_link(name="ogbl-collab")
        if task == "NodeClassification":
            if dataset in ["ogbn-arxiv"]:
                self.evaluator_class = eval_class(name=dataset)
            else:
                self.evaluator_class = eval_class(name="ogbn-arxiv")

    def hits_k(self, y_pred_pos, y_pred_neg, K=50):
        self.evaluator_link.K = K
        return self.evaluator_link.eval({"y_pred_pos": y_pred_pos, "y_pred_neg": y_pred_neg})

    def mrr(self, y_pred_pos, y_pred_neg):
        return (
            self.evaluator_link.eval({"y_pred_pos": y_pred_pos, "y_pred_neg": y_pred_neg})["mrr_list"]
            .mean()
            .item()
        )

    def ogb_acc(self, y, y_hat):
        return self.evaluator_class.eval({"y_true": y, "y_pred": y_hat})

    def accuracy(self, y, y_hat):
        return accuracy_score(y_true=y, y_pred=y_hat)

    def roc_auc(self, y, y_hat):
        return roc_auc_score(y_true=y, y_score=y_hat)

    def f1(self, y, y_hat, avg):
        return f1_score(y_true=y, y_pred=y_hat, average=avg)

    def precision(self, y, y_hat, avg):
        return precision_score(y_true=y, y_pred=y_hat, average=avg)

    def recall(self, y, y_hat, avg):
        return recall_score(y_true=y, y_pred=y_hat, average=avg)

    def auc_metric(self, y, y_hat):
        fpr, tpr, thresholds = roc_curve(y, y_hat, pos_label=1)
        return auc(fpr, tpr)

    def collect_metrics(self, predictions: dict):
        """
        Example:
        predictions = {'train':{
        'y_pred_pos':torch.rand((100,1)),
        'y_pred_neg':torch.rand((100,1))},
        'valid':{
        'y_pred_pos':torch.rand((100,1)),
        'y_pred_neg':torch.rand((100,1))},
        'test':{
        'y_pred_pos':torch.rand((100,1)),
        'y_pred_neg':torch.rand((100,1))}
        }
        """
        results = {}
        for data_type in predictions.keys():
            inner_results = {}

            if self.task == "LinkPrediction" and "mrr" not in self.metrics_list:
                y_hat = torch.cat(
                    [predictions[data_type]["y_pred_pos"], predictions[data_type]["y_pred_neg"]], dim=0
                )
                y_hat_raw = copy.deepcopy(y_hat)
                y_hat = (y_hat >= 0.5).float().detach().cpu().numpy()
                y_true = torch.cat(
                    [
                        torch.ones(predictions[data_type]["y_pred_pos"].shape[0]),
                        torch.zeros(predictions[data_type]["y_pred_neg"].shape[0]),
                    ]
                ).numpy()
            elif self.task == "NodeClassification":
                y_hat = predictions[data_type]["y_hat"].cpu()
                y_true = predictions[data_type]["y_true"].cpu()

            for metric in self.metrics_list:
                if metric == "f1-micro":
                    inner_results[metric] = self.f1(y=y_true, y_hat=y_hat, avg="micro")
                elif metric == "acc":
                    inner_results[metric] = self.accuracy(y=y_true, y_hat=y_hat)
                elif metric == "roc_auc":
                    inner_results[metric] = self.roc_auc(y=y_true, y_hat=y_hat_raw.detach().cpu().numpy())
                elif metric == "precision-micro":
                    inner_results[metric] = self.precision(y=y_true, y_hat=y_hat, avg="micro")
                elif metric == "recall-micro":
                    inner_results[metric] = self.recall(y=y_true, y_hat=y_hat, avg="micro")
                elif metric == "auc":
                    inner_results[metric] = self.auc_metric(y=y_true, y_hat=y_hat)
                elif metric == "f1-macro":
                    inner_results[metric] = self.f1(y=y_true, y_hat=y_hat, avg="macro")
                elif metric == "recall-macro":
                    inner_results[metric] = self.recall(y=y_true, y_hat=y_hat, avg="macro")
                elif metric == "precision-macro":
                    inner_results[metric] = self.precision(y=y_true, y_hat=y_hat, avg="macro")
                elif "hits" in metric:
                    K = int((metric.split("@"))[1])
                    inner_results[metric] = self.hits_k(
                        K=K,
                        y_pred_pos=predictions[data_type]["y_pred_pos"],
                        y_pred_neg=predictions[data_type]["y_pred_neg"],
                    )[metric]
                elif metric == "rocauc":
                    inner_results[metric] = self.roc_auc(
                        y_pred_pos=predictions[data_type]["y_pred_pos"],
                        y_pred_neg=predictions[data_type]["y_pred_neg"],
                    )
                elif metric == "mrr":
                    inner_results[metric] = self.mrr(
                        y_pred_pos=predictions[data_type]["y_pred_pos"],
                        y_pred_neg=predictions[data_type]["y_pred_neg"],
                    )

            results[data_type] = inner_results

        
        return results
