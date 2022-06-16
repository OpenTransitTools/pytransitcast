import asyncio
import datetime
import json
import logging as log
import model_records
import multiprocessing as mp
import numpy as np
import os
import queue
import time
import xgboost as xgb

from nats.aio.client import Client as NATS
from nats.errors import TimeoutError
from typing import NamedTuple
from typing import Union


class InferenceRequest(NamedTuple):
    """An inference request as given to this runner by an external source"""

    request_id: str
    ml_model_id: int
    features: np.ndarray


class InferenceResponse(NamedTuple):
    """A response given from this module via NATS"""

    request_id: str
    timestamp: int
    prediction: str
    error: str


class Job(NamedTuple):
    """A job for a subprocess to complete, containing the model to run inference on and the input data"""

    model: xgb.XGBRegressor
    nats_host: str
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
        new_loaded_models = {}

        # TODO: only load relevant models
        # By setting this to tomorrow and setting train flag to false we get all models that are currently trained
        tomorrow = datetime.date.today() + datetime.timedelta(days=1)
        models = model_records.get_current_models(self.conn, tomorrow, False)  # type: ignore

        for model in models:
            loaded = xgb.XGBRegressor()
            model_bytes = bytearray(model.model_blob)
            # TODO: might it be more efficient to pickle things here?
            loaded.load_model(model_bytes)
            new_loaded_models[str(model.ml_model_id)] = loaded
        self.loaded_models = new_loaded_models
        log.info(f"Loaded {len(self.loaded_models)} models")

    def add_job(self, request: InferenceRequest, nats_host: str):
        try:
            loaded_model = self.loaded_models[request.ml_model_id]
        except Exception as e:
            log.warn(f"Model {request.ml_model_id} couldn't be loaded: {e}")
            return
        self.queue.put(Job(loaded_model, nats_host, request))

    def tear_down(self):
        log.info("Tearing down")
        self.pool.close()
        self.pool.join()


def queue_handler(queue: queue.Queue):
    """The loop that runs in the subprocess"""
    log.info(f"Launching thread {os.getpid()}")
    while True:
        job = queue.get()
        log.info(
            f"Thread {os.getpid()} starting new job {job.inference_request.request_id}"
        )
        infer(job)


def infer(job: Job):
    """The method run by the subprocess which does the correct inference"""

    async def send_msg(r: InferenceResponse):
        nc = NATS()
        await nc.connect(job.nats_host)
        await nc.publish(
            "transitcast_inference_response", json.dumps(r._asdict()).encode()
        )
        log.info(f"Reponse for request {job.inference_request.request_id} sent")

    response: Union[InferenceResponse, None] = None

    try:
        result = job.model.predict(job.inference_request.features)
        if len(result) < 1:
            raise Exception("Invalid result returned from model")
        response = InferenceResponse(
            job.inference_request.request_id,
            int(time.time()),
            str(result[0]),
            "",
        )
    except Exception as e:
        log.error(f"Couldn't infer: {e}")
        response = InferenceResponse(
            job.inference_request.request_id,
            int(time.time()),
            "",
            str(e),
        )
    finally:
        if response is not None:
            asyncio.run(send_msg(response))


async def start_nats_listener(nats_host: str, runner):
    nc = NATS()

    async def nats_message_handler(message):
        msg = json.loads(message.data.decode())
        try:
            request = InferenceRequest(
                msg["request_id"], msg["ml_model_id"], np.array([msg["features"]])
            )
            log.info(
                f"Recieved inference request with id {msg['request_id']} created at {datetime.datetime.fromtimestamp(msg['timestamp'])}"
            )
            runner.add_job(
                request,
                nats_host,
            )
        except Exception as e:
            log.error(f"Invalid request recieved: {msg}")
            await nc.publish(
                "transitcast_inference_response",
                json.dumps({"error": f"Invalid request recieved: {e}"}).encode(),
            )

    try:
        await nc.connect(nats_host)
        await nc.subscribe("transitcast_inference_request", cb=nats_message_handler)
    except TimeoutError:
        log.error(f"Could not connect to NATS: Request to {nats_host} timed out")


def start_runner(conn, runner_process_count: int, nats_host: str):
    runner = ModelRunner(conn, runner_process_count)

    try:
        # TODO: run this method periodically, possibly in its own thread
        runner.load_relevant_models()

        loop = asyncio.get_event_loop()
        loop.run_until_complete(start_nats_listener(nats_host, runner))
        loop.run_forever()
        loop.close()

    except Exception as e:
        log.error(f"An unhandled error has occured while loading models: {e}")
    finally:
        log.info(f"Closing threads...")
        runner.tear_down()
