import argparse
import math

import yaml
from transformers import Trainer

from protein_lm.modeling.getters.data_collator import get_data_collator
from protein_lm.modeling.getters.dataset import get_dataset
from protein_lm.modeling.getters.model import get_model
from protein_lm.modeling.getters.tokenizer import get_tokenizer
from protein_lm.modeling.getters.training_args import get_training_args


def train(
    config_file: str,
):
    """
    Main script to train APT.
    """
    with open(config_file, "r") as cf:
        config_dict = yaml.safe_load(cf)
        print(config_dict)

    tokenizer = get_tokenizer(config_dict=config_dict["tokenizer"])

    train_ds = get_dataset(
        config_dict=config_dict["dataset"],
        tokenizer=tokenizer,
    )

    model = get_model(
        config_dict=config_dict["model"],
    )
    model.train()

    data_collator = get_data_collator(
        config_dict=config_dict["data_collator"],
    )

    training_args = get_training_args(
        config_dict=config_dict["training_args"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=data_collator,
    )

    train_result = trainer.train()
    trainer.save_model()  # Saves the tokenizer too for easy upload
    metrics = train_result.metrics
    try:
        perplexity = math.exp(metrics["train_loss"])
    except OverflowError:
        perplexity = float("inf")
    metrics["perplexity"] = perplexity
    print("metrics:", metrics)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        default="protein_lm/configs/train/toy_localcsv.yaml",
        type=str,
        help="Config yaml for training",
    )
    args = parser.parse_args()

    train(config_file=args.config_file)
