import warnings
import logging as log
import datetime
import data_loader
import model_trainer
from config import load_config

# Ignore pandas.Int64Index warnings internal to xgboost.
warnings.filterwarnings('ignore', 'pandas.Int64Index', FutureWarning, 'xgboost')

log.basicConfig(level=log.INFO, format='%(asctime)s:%(levelname)s:%(message)s')


def main():
    config = load_config()
    log.info(f"Using config: {config}")

    start_date = datetime.datetime.strptime('2021-01-01', '%Y-%m-%d')
    now = datetime.datetime.now()
    reattempt_date = now - datetime.timedelta(days=config.reattempt_training_days)
    end_date = now.replace(hour=0, minute=0, second=0, microsecond=0)

    conn = data_loader.connect(config)
    model_trainer.train_pending_models(conn, start_date, end_date, reattempt_date)


if __name__ == "__main__":
    main()
