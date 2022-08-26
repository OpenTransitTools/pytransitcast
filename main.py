import argparse
import warnings
import logging as log
import datetime
import data_loader
import model_trainer
import model_runner
import config

# Ignore pandas.Int64Index warnings internal to xgboost.
warnings.filterwarnings("ignore", "pandas.Int64Index", FutureWarning, "xgboost")

log.basicConfig(level=log.INFO, format="%(asctime)s:%(levelname)s:%(message)s")


def train(cfg: config.Config):
    conn = data_loader.connect(cfg)
    start_date = datetime.datetime.strptime("2021-01-01", "%Y-%m-%d")
    now = datetime.datetime.now()
    reattempt_date = now - datetime.timedelta(days=cfg.reattempt_training_days)
    end_date = now.replace(hour=0, minute=0, second=0, microsecond=0)

    model_trainer.train_pending_models(conn, start_date, end_date, reattempt_date)


def infer(cfg: config.Config):
    model_runner.start_runner(cfg)


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

    cfg = config.load_config()
    log.info(f"Using config: {cfg}")

    if args.mode == "train":
        train(cfg)
    if args.mode == "infer":
        infer(cfg)
