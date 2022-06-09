from typing import NamedTuple
import datetime
import logging as log
import model_records
import os
import queue
import xgboost as xgb
import numpy as np
import multiprocessing as mp


class InferenceRequest(NamedTuple):
    request_id: str
    timestamp: int
    features: np.ndarray


class Job(NamedTuple):
    model: xgb.XGBRegressor
    inference_request: InferenceRequest


class ModelRunner:
    """Starts worker subprocesses, handles incoming jobs."""

    def __init__(self, conn, thread_count: int):
        self.conn = conn
        self.loaded_models = {}
        self.queue = mp.Queue()

        self.pool = mp.Pool(
            thread_count,
            queue_handler,
            (self.queue,),
        )

    def load_relevant_models(self):
        # TODO: only load relevant models
        # By setting this to tomorrow and setting train flag to false we get all models that are currently trained
        tomorrow = datetime.date.today() + datetime.timedelta(days=1)
        models = model_records.get_current_models(self.conn, tomorrow, False)  # type: ignore
        for model in models:

            loaded = xgb.XGBRegressor()
            model_bytes = bytearray(model.model_blob)
            loaded.load_model(model_bytes)
            self.loaded_models[str(model.ml_model_id)] = loaded
        log.info(f"Loaded {len(self.loaded_models)} models")

    def add_job(self, model_id: str, request: InferenceRequest):
        try:
            loaded_model = self.loaded_models[model_id]
        except Exception as e:
            log.warn(f"Model {model_id} couldn't be loaded: {e}")
            return
        self.queue.put(Job(loaded_model, request))

    def tear_down(self):
        log.info("Tearing down")
        self.pool.close()
        self.pool.join()


def queue_handler(queue: queue.Queue):
    """The loop that runs in the subprocess"""
    log.info(f"Launching thread {os.getpid()}")
    while True:
        job = queue.get()
        log.info(f"Thread {os.getpid()} starting new job")
        infer(job)


def infer(job: Job):
    """The method run by the subprocess which does the correct inference"""

    try:
        result = job.model.predict(job.inference_request.features)
        print(result)
    except Exception as e:
        log.error(f"Couldn't infer: {e}")


def start_runner(conn, runner_process_count: int):
    runner = ModelRunner(conn, runner_process_count)

    try:
        runner.load_relevant_models()
        for i in range(0, 415):
            runner.add_job(
                str(i),
                InferenceRequest(
                    "0", 0, np.array([[5, 1, 12, 20, 50, 0, 0, 0, 100, 40, 0, 0]])
                ),
            )
    except Exception as e:
        log.error(f"An unhandled error has occured: {e}")
    finally:
        log.info(f"Closing threads...")
        runner.tear_down()
