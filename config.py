import os
import logging as log


class Config(object):
    """Configuration parameters"""

    def __init__(
        self,
        db_user: str,
        db_password: str,
        db_name: str,
        db_host: str,
        reattempt_training_days: int,
        runner_process_count: int,
        nats_host: str,
    ):
        self.db_user = db_user
        self.db_password = db_password
        self.db_name = db_name
        self.db_host = db_host
        self.reattempt_training_days = reattempt_training_days
        self.runner_process_count = runner_process_count
        self.nats_host = nats_host

    def __str__(self):
        copy = self.__dict__.copy()
        copy['db_password'] = '<masked>'
        return str(copy)


def load_config() -> Config:
    """loads configuration from environment variables.
    Reads from variables
    PYTRANSITCAST_DB_USER, PYTRANSITCAST_DB_PASSWORD, PYTRANSITCASTR_DB_NAME and PYTRANSITCAST_DB_HOST"""
    try:
        return Config(
            db_user=os.environ["PYTRANSITCAST_DB_USER"],
            db_password=os.environ["PYTRANSITCAST_DB_PASSWORD"],
            db_name=os.environ["PYTRANSITCAST_DB_NAME"],
            db_host=os.environ["PYTRANSITCAST_DB_HOST"],
            reattempt_training_days=get_int_environment_variable(
                "PYTRANSITCAST_REATTEMPT_TRAINING_DAYS", 30
            ),
            runner_process_count=get_int_environment_variable(
                "PYTRANSITCAST_RUNNER_PROCESS_COUNT", 4
            ),
            nats_host=os.environ["PYTRANSITCAST_NATS_HOST"],
        )
    except KeyError as e:
        log.warning(f"Unable to load configuration, missing parameters %{e}")
        exit(1)


def get_int_environment_variable(name: str, default: int) -> int:
    if name not in os.environ:
        return default
    try:
        return int(os.environ[name])
    except ValueError as e:
        log.warning(f"Unable to read {name} environment variable as int using default:{default}")
    return default
