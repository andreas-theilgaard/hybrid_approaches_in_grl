from src.models.NodeClassification.combined.logits_integration import fit_logits_integration
from src.models.NodeClassification.combined.combined_latents import fit_combined_latents


def fit_combined_class(config, dataset, training_args, Logger, log, seeds, save_path):
    if config.dataset.combined.type == "LogitsIntegration":
        fit_logits_integration(
            config=config,
            dataset=dataset,
            training_args=training_args,
            Logger=Logger,
            log=log,
            seeds=seeds,
            save_path=save_path,
        )
    elif config.dataset.combined.type == "CombinedLatents":
        fit_combined_latents(
            config=config,
            dataset=dataset,
            training_args=training_args,
            Logger=Logger,
            log=log,
            seeds=seeds,
            save_path=save_path,
        )

