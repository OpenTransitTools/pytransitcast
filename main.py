import argparse
import warnings
import logging as log
import datetime
import data_loader
import model_trainer
import model_runner
from config import load_config

# Ignore pandas.Int64Index warnings internal to xgboost.
warnings.filterwarnings("ignore", "pandas.Int64Index", FutureWarning, "xgboost")

log.basicConfig(level=log.INFO, format="%(asctime)s:%(levelname)s:%(message)s")


def train(config):
    conn = data_loader.connect(config)
    start_date = datetime.datetime.strptime("2021-01-01", "%Y-%m-%d")
    now = datetime.datetime.now()
    reattempt_date = now - datetime.timedelta(days=config.reattempt_training_days)
    end_date = now.replace(hour=0, minute=0, second=0, microsecond=0)

    conn = data_loader.connect(config)
    model_trainer.train_pending_models(conn, start_date, end_date, reattempt_date)


def infer(config):
    conn = data_loader.connect(config)
    model_runner.start_runner(
        conn, config.runner_process_count, config.nats_host, config.rmse_margin
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run or train pytransitcast ML Models")

    parser.add_argument(
        "--mode",
        dest="mode",
        choices=["infer", "train"],
        required=True,
        help="Whether to train models or run the inference server. Must be one of 'train' or 'infer'",
    )

    args = parser.parse_args()

    config = load_config()
    log.info(f"Using config: {config}")

    if args.mode == "train":
        train(config)
    if args.mode == "infer":
        infer(config)
