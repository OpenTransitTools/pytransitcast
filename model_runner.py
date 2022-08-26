import asyncio
import datetime
import json
import logging as log
import pickle
import signal
import sys

import config
import data_loader
import model_records
import multiprocessing as mp
import numpy as np
import os
import time
import xgboost as xgb

from nats.aio.client import Client as NATS
from nats.aio.msg import Msg
from typing import NamedTuple, Dict
from typing import Union


class InferenceRequest(NamedTuple):
    """An inference request as given to this runner by an external source"""

    request_id: str
    ml_model_id: int
    version: int
    features: np.ndarray
    timestamp: int


class InferenceResponse(NamedTuple):
    """A response given from this module via NATS"""

    request_id: str
    timestamp: int
    prediction: float
    error: str


class ModelCollection:
    """Holds models for a child process."""

    def __init__(self):
        self.loaded_models = {}

    def put_models(self, new_loaded_models: Dict[str, xgb.XGBRegressor]):
        self.loaded_models = new_loaded_models

    def get_model(self, request: InferenceRequest):
        try:
            return self.loaded_models[
                str(request.ml_model_id) + "_" + str(request.version)
                ]
        except Exception as e:
            raise Exception(
                f"Model {request.ml_model_id} version {request.version} couldn't be loaded: {e}"
            )


def load_xgboost_models(ml_models: Dict[str, model_records.MLModel]):
    """loads XGBRegressor from model_records.MLModel.model_blob json value"""
    results = {}
    for key, value in ml_models.items():
        loaded_model = xgb.XGBRegressor()
        model_bytes = bytearray(value.model_blob)
        loaded_model.load_model(model_bytes)
        results[key] = loaded_model
    return results


class ChildProcessWrapper:
    """ChildProcessWrapper holds the child process.
    Class is intended to hold multiprocessing objects like Queue's when needed (for reloading models for example)"""

    def __init__(self,
                 process: mp.Process):
        self.process = process


class ChildrenProcesses:
    """ChildrenProcesses holds children processes owned by parent runner"""

    def __init__(self):
        self.children = []
        self.process_ready_queue = mp.Queue()

    def add_child(self, child: ChildProcessWrapper):
        self.children.append(child)

    def start_processes(self):
        log.info("starting child processes")
        count = 0
        for _, child in enumerate(self.children):
            child.process.start()
            count += 1
            log.info(f"{count} children initialized")

        waiting_readied_processes = len(self.children)

        while waiting_readied_processes > 0:
            log.info(f"waiting for {waiting_readied_processes} child processes to finish initialization")
            self.process_ready_queue.get()
            waiting_readied_processes -= 1

        log.info(f"{len(self.children)} children processes initialized")

    def terminate_children(self):
        for _, child in enumerate(self.children):
            pid = child.process.pid
            if pid is not None:
                log.info(f"terminating child {pid}")
                child.process.kill()


def create_child_processes(cfg: config.Config, children_processes: ChildrenProcesses):
    """load models, slice them into buckets from cfg.inference_buckets, creates children processes for each
    and adds them to children_processes"""
    model_buckets = load_models(cfg)

    for bucket_number in range(0, cfg.inference_buckets):

        for bucket_process_number in range(0, cfg.processes_per_bucket):
            process = mp.Process(target=start_inference_child_process,
                                 args=(cfg,
                                       children_processes.process_ready_queue,
                                       bucket_number,
                                       bucket_process_number,
                                       model_buckets[bucket_number]))
            children_processes.add_child(ChildProcessWrapper(process))


def load_models(cfg: config.Config):
    """Loads models optionally from cache or database if cache is not configured or not present."""

    if cfg.use_cached_models and os.path.exists(cfg.cached_models_path):
        log.info(f"Loading model from cache")
        file = open(cfg.cached_models_path, "rb")
        model_buckets = pickle.load(file)
        file.close()
        return model_buckets

    log.info("No model cache")
    log.info(f"Establishing database connection and loading models")
    conn = data_loader.connect(cfg)
    model_buckets = load_uncached_relevant_models(conn, cfg.rmse_margin, cfg.inference_buckets)
    log.info(f"Done loading models, closing database connection")
    conn.close()

    if cfg.use_cached_models:
        log.info("Saving model cache")
        file = open(cfg.cached_models_path, 'wb')
        pickle.dump(model_buckets, file)
        file.close()
    return model_buckets


def load_uncached_relevant_models(conn, rmse_margin: int, buckets: int):
    """Loads models from database, sorting them into array of buckets by name"""
    results = []

    # initialize result buckets
    for _ in range(0, buckets):
        results.append({})

    def add_ml_model(ml_model: model_records.MLModel):
        bucket_number = ml_model.ml_model_id % buckets
        results[bucket_number][
            str(ml_model.ml_model_id) + "_" + str(ml_model.version)
            ] = ml_model

    count = 0
    for bucket in enumerate(results):
        count += len(bucket)

    model_records.get_relevant_models(conn, rmse_margin, add_ml_model)
    log.info(f"Loaded {count} models for {buckets} buckets")
    return results


def setup_child_signals(process_name: str):
    """listen for signals from a child inference process"""

    def signal_handler(sig_number: int, frame):
        log.info(f"{process_name} exiting on signal {sig_number}")
        sys.exit(0)

    signal.signal(signal.SIGHUP, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def start_inference_child_process(cfg: config.Config,
                                  process_ready_queue: mp.Queue,
                                  bucket_number: int,
                                  bucket_process_number: int,
                                  models: Dict[str, model_records.MLModel]):
    """entry point for inference child processes"""
    process_name = f"process_{bucket_number}_{bucket_process_number}_pid:{os.getpid()}"
    setup_child_signals(process_name)
    try:
        log.info(f"{process_name} converting model records to xgboost models")
        models = load_xgboost_models(models)
        log.info(f"{process_name} contains {len(models)} Models")
        model_collection = ModelCollection()
        model_collection.put_models(models)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(run_inference_queue(process_name=process_name,
                                                    model_collection=model_collection,
                                                    process_ready_queue=process_ready_queue,
                                                    nats_host=cfg.nats_host,
                                                    bucket_number=bucket_number))
        loop.run_forever()
    except KeyboardInterrupt:
        log.info(f"{process_name} exiting on keyboard interrupt")


async def run_inference_queue(process_name: str,
                              model_collection: ModelCollection,
                              process_ready_queue: mp.Queue,
                              nats_host: str,
                              bucket_number: int):
    """listen for inference requests for child process bucket and service request"""
    try:
        nc = NATS()
        await nc.connect(nats_host)
        sub = await nc.subscribe(f"inference-request.{bucket_number}", queue="runner")
        log.info(f"{process_name} ready")
        process_ready_queue.put(True)
        async for message in sub.messages:
            await service_inference_request(process_name=process_name,
                                            nc=nc,
                                            model_collection=model_collection,
                                            message=message)

    except TimeoutError:
        log.error(f"Could not connect to NATS: Request to {nats_host} timed out")


async def service_inference_request(process_name: str,
                                    nc: NATS,
                                    model_collection: ModelCollection,
                                    message: Msg):
    """decode nats Msg and perform inference, catch and log any exceptions"""
    msg = None
    try:
        msg = json.loads(message.data.decode())
        inference_request = InferenceRequest(
            request_id=msg["request_id"],
            ml_model_id=msg["ml_model_id"],
            version=msg["version"],
            features=np.array([msg["features"]]),
            timestamp=msg["timestamp"]
        )
        await infer(process_name=process_name,
                    inference_request=inference_request,
                    nc=nc,
                    model_collection=model_collection)
    except Exception as e:
        error_message = f"error servicing request: {msg}"
        log.error(error_message)
        log.error(e)
        try:
            await nc.publish(
                "inference-response",
                json.dumps({"error": error_message}).encode(),
            )
        except Exception as e2:
            log.error("Unable to publish error response")
            log.error(e2)


async def infer(process_name: str,
                inference_request: InferenceRequest,
                nc: NATS,
                model_collection: ModelCollection):
    """perform inference and send results"""

    response: Union[InferenceResponse, None] = None

    try:
        model = model_collection.get_model(inference_request)
        result = model.predict(inference_request.features)

        if len(result) < 1:
            raise Exception("Invalid result returned from model")
        response = InferenceResponse(
            inference_request.request_id,
            int(time.time()),
            result[0].astype(float),
            "",
        )
    except Exception as e:
        log.error(
            f"Error processing request {inference_request.request_id}: Couldn't infer: {e}"
        )
        response = InferenceResponse(
            inference_request.request_id,
            int(time.time()),
            0.0,
            str(e),
        )
    finally:
        if response is not None:
            took = int(time.time()) - inference_request.timestamp
            log.info(
                f"{process_name} processed inference request with id {inference_request.request_id} "
                f"created at {datetime.datetime.fromtimestamp(inference_request.timestamp)} took:{took}"
            )
            await nc.publish(
                "inference-response", json.dumps(response._asdict()).encode()
            )
            await nc.flush(timeout=2)


def setup_signals(child_processes: ChildrenProcesses):
    """listen for signals to parent process, terminate children and exit"""
    def signal_handler(sig_number: int, frame):
        log.info(f"exiting on signal {sig_number}")
        child_processes.terminate_children()
        sys.exit(0)

    signal.signal(signal.SIGHUP, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def main_loop():
    while True:
        time.sleep(5)
        # TODO:: add maintenance here (inform children to reload models)


def start_runner(cfg: config.Config):
    try:
        child_processes = ChildrenProcesses()
        setup_signals(child_processes)
        log.info(f"parent pid:{os.getpid()} initializing inference processes")
        create_child_processes(cfg, child_processes)
        child_processes.start_processes()
        log.info(f"parent pid:{os.getpid()} children ready")
        main_loop()
    except KeyboardInterrupt:
        log.info("main process exiting on keyboard interrupt")
    except Exception as e:
        log.error(f"An unhandled error has occurred while starting runner: {e}")
