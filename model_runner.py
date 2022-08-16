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
from multiprocessing.connection import Pipe, wait, Connection
from typing import NamedTuple
from threading import Thread
from typing import Union


class InferenceRequest(NamedTuple):
    """An inference request as given to this runner by an external source"""

    request_id: str
    ml_model_id: int
    version: int
    features: np.ndarray


class InferenceResponse(NamedTuple):
    """A response given from this module via NATS"""

    request_id: str
    timestamp: int
    prediction: float
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
        self.nc = NATS()
        self.response_queue = mp.Queue()

        self.pool = mp.Pool(
            thread_count,
            queue_handler,
            (self.queue, self.response_queue,),
        )

    async def send_nats_msgs(self):
        while True:
            response = self.response_queue.get()
            await self.nc.publish("inference-response", json.dumps(response._asdict()).encode())
            log.info(f"Response for request {response.request_id} sent")

    def load_relevant_models(self, rmse_margin: int):
        new_loaded_models = {}

        models = model_records.get_relevant_models(self.conn, rmse_margin)

        for model in models:
            loaded = xgb.XGBRegressor()
            model_bytes = bytearray(model.model_blob)
            # TODO: might it be more efficient to pickle things here?
            loaded.load_model(model_bytes)
            new_loaded_models[
                str(model.ml_model_id) + "_" + str(model.version)
            ] = loaded
        self.loaded_models = new_loaded_models
        log.info(f"Loaded {len(self.loaded_models)} models")

    def add_job(self, request: InferenceRequest, nats_host: str):
        try:
            loaded_model = self.loaded_models[
                str(request.ml_model_id) + "_" + str(request.version)
            ]
        except Exception as e:
            raise Exception(
                f"Model {request.ml_model_id} version {request.version} couldn't be loaded: {e}"
            )
        new_job = Job(loaded_model, nats_host, request)
        self.queue.put(new_job)

    def tear_down(self):
        log.info("Tearing down")
        self.pool.close()
        self.pool.join()


def queue_handler(queue: queue.Queue, response_queue: queue.Queue):
    """The loop that runs in the subprocess"""
    log.info(f"Launching thread {os.getpid()}")
    while True:
        job = queue.get()
        log.info(
            f"Thread {os.getpid()} starting new job {job.inference_request.request_id}"
        )
        response = infer(job)
        response_queue.put(response)


def infer(job: Job):
    """The method run by the subprocess which does the correct inference"""

    response: Union[InferenceResponse, None] = None

    try:
        result = job.model.predict(job.inference_request.features)
        if len(result) < 1:
            raise Exception("Invalid result returned from model")
        response = InferenceResponse(
            job.inference_request.request_id,
            int(time.time()),
            result[0].astype(float),
            "",
        )
        # time.sleep(90)
    except Exception as e:
        log.error(
            f"Error processing request {job.inference_request.request_id}: Couldn't infer: {e}"
        )
        response = InferenceResponse(
            job.inference_request.request_id,
            int(time.time()),
            0.0,
            str(e),
        )
    finally:
        if response is not None:
            return response


async def start_nats_listener(nats_host: str, runner):
    async def nats_message_handler(message):
        msg = json.loads(message.data.decode())
        try:
            request = InferenceRequest(
                msg["request_id"],
                msg["ml_model_id"],
                msg["version"],
                np.array([msg["features"]]),
            )
            log.info(
                f"Received inference request with id {msg['request_id']} created at {datetime.datetime.fromtimestamp(msg['timestamp'])}"
            )
            runner.add_job(
                request,
                nats_host,
            )
        except Exception as e:
            log.error(f"Invalid request received: {msg}")
            log.error(e)
            await runner.nc.publish(
                "inference-response",
                json.dumps(
                    {"error": f"Invalid request received: {e}"}).encode(),
            )

    try:
        await runner.nc.connect(nats_host)
        await runner.nc.subscribe("inference-request", cb=nats_message_handler)
    except TimeoutError:
        log.error(
            f"Could not connect to NATS: Request to {nats_host} timed out")


def nats_sender_entrypoint(runner):
    asyncio.run(runner.send_nats_msgs())


def start_runner(conn, runner_process_count: int, nats_host: str, rmse_margin: int):
    runner = ModelRunner(conn, runner_process_count)

    try:
        # TODO: run this method periodically, possibly in its own thread
        runner.load_relevant_models(rmse_margin)

        t = Thread(target=nats_sender_entrypoint,
                   args=(runner, ), daemon=True)
        t.start()

        loop = asyncio.get_event_loop()
        loop.create_task(start_nats_listener(nats_host, runner))
        loop.run_forever()
        loop.close()

    except Exception as e:
        log.error(f"An unhandled error has occurred while loading models: {e}")
    finally:
        log.info(f"Closing threads...")
        runner.tear_down()
