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
        inference_buckets: int,
        processes_per_bucket: int,
        nats_host: str,
        rmse_margin: int,
        use_cached_models: bool,
        cached_models_path: str,
    ):
        self.db_user = db_user
        self.db_password = db_password
        self.db_name = db_name
        self.db_host = db_host
        self.reattempt_training_days = reattempt_training_days
        self.inference_buckets = inference_buckets
        self.nats_host = nats_host
        self.rmse_margin = rmse_margin
        self.processes_per_bucket = processes_per_bucket
        self.use_cached_models = use_cached_models
        self.cached_models_path = cached_models_path

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
            inference_buckets=get_int_environment_variable(
                "PYTRANSITCAST_INFERENCE_BUCKETS", 8
            ),
            nats_host=os.environ["PYTRANSITCAST_NATS_HOST"],
            rmse_margin=get_int_environment_variable("PYTRANSITCAST_RMSE_MARGIN", 0),
            processes_per_bucket=get_int_environment_variable("PYTRANSITCAST_PROCESSES_PER_BUCKET", 3),
            use_cached_models=get_bool_environment_variable("PYTRANSITCAST_USE_CACHED_MODELS", False),
            cached_models_path=get_str_environment_variable("PYTRANSITCAST_CACHE_MODELS_PATH", "models.cache")
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


def get_bool_environment_variable(name: str, default: bool) -> bool:
    if name not in os.environ:
        return default
    try:
        return bool(os.environ[name])
    except ValueError as e:
        log.warning(f"Unable to read {name} environment variable as int using default:{default}")
    return default


def get_str_environment_variable(name: str, default: str) -> str:
    if name not in os.environ:
        return default
    try:
        return os.environ[name]
    except ValueError as e:
        log.warning(f"Unable to read {name} environment variable as int using default:{default}")
    return default
