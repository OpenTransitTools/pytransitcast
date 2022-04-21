from datetime import datetime

import logging as log
from typing import Dict, List


class MLModelType:
    """Holds records from ml_model_type table, each row is a type of model"""
    def __init__(self,
                 ml_model_type_id: int,
                 name: str):
        self.ml_model_type_id = ml_model_type_id
        self.name = name


class MLModelStops:
    """Holds records from ml_model_stops table, each row is an instance of stop to stop movement to be trained with"""
    def __init__(self, ml_model_stop_id: int,
                 ml_model_id: int,
                 sequence: int,
                 stop_id: str,
                 next_stop_id: str):
        self.ml_model_stop_id = ml_model_stop_id
        self.ml_model_id = ml_model_id
        self.sequence = sequence
        self.stop_id = stop_id
        self.next_stop_id = next_stop_id
        self.model_name = ml_model_stops_name([self])


def ml_model_stops_name(model_stop_list: [MLModelStops]) -> str:
    """create name of model by joining stops ids with an underscore"""
    results = []
    for ms in model_stop_list:
        if len(results) == 0:
            results.append(ms.stop_id)
        results.append(ms.next_stop_id)
    return '_'.join(results)


class MLModel:
    """Holds records from ml_Model table, each row is an instance of a model to be trained"""
    def __init__(self,
                 ml_model_id: int,
                 version: int,
                 start_timestamp: datetime,
                 end_timestamp: datetime,
                 ml_model_type_id: int,
                 train_flag: bool,
                 trained_timestamp: datetime,
                 avg_rmse: float,
                 ml_rmse: float,
                 feature_trained_start_timestamp: datetime,
                 feature_trained_end_timestamp: datetime,
                 model_name: str,
                 currently_relevant: bool,
                 last_train_attempt_timestamp: datetime,
                 observed_stop_count: int,
                 median: float,
                 average: float,
                 model_blob: bytes):
        self.ml_model_id = ml_model_id
        self.version = version
        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp
        self.ml_model_type_id = ml_model_type_id
        self.train_flag = train_flag
        self.trained_timestamp = trained_timestamp
        self.avg_rmse = avg_rmse
        self.ml_rmse = ml_rmse
        self.feature_trained_start_timestamp = feature_trained_start_timestamp
        self.feature_trained_end_timestamp = feature_trained_end_timestamp
        self.model_name = model_name
        self.currently_relevant = currently_relevant
        self.last_train_attempt_timestamp = last_train_attempt_timestamp
        self.observed_stop_count = observed_stop_count
        self.median = median
        self.average = average
        self.model_blob = model_blob
        self.model_stops = [MLModelStops]


def get_ml_model_type(source_conn, name: str):
    """with source connection retrieve the ml_model_type with name"""
    cursor = source_conn.cursor()
    query = """select ml_model_type_id, name
    from ml_model_type where name = %s"""
    log.info(f"Issuing query `{query}` with args {name}")
    cursor.execute(query, [name])
    row = cursor.fetchone()
    cursor.close()
    if row is None:
        return None
    return MLModelType(row[0], row[1])


def _get_ml_model_stop_dict(source_conn,
                            reattempt_date: datetime,
                            train_flag: bool) -> Dict[int, List[MLModelStops,]]:
    """Get dictionary of MLModelStops list by ml_model_id"""
    cursor = source_conn.cursor()
    query = """select mms.ml_model_stop_id, mms.ml_model_id, mms.sequence, mms.stop_id, mms.next_stop_id 
    from ml_model_stop mms, ml_model mm
where mm.ml_model_id = mms.ml_model_id
and mm.train_flag = %s
and (mm.last_train_attempt_timestamp is null or mm.last_train_attempt_timestamp < %s)
order by mms.ml_model_id, mms.sequence"""
    log.info(f"Issuing query `{query}` with args {train_flag}, '{reattempt_date}'")
    cursor.execute(query, [train_flag, reattempt_date])
    results = {}
    row = cursor.fetchone()
    while row is not None:
        model_stops = MLModelStops(row[0], row[1], row[2], row[3], row[4])
        if model_stops.ml_model_id not in results:
            results[model_stops.ml_model_id] = []
        results[model_stops.ml_model_id].append(model_stops)
        row = cursor.fetchone()

    cursor.close()

    return results


def get_current_models(conn, reattempt_date: datetime, train_flag: bool) -> [MLModel]:
    """retrieves all current MLModels that need to be trained or not according to train_flag argument"""
    model_stop_dict = _get_ml_model_stop_dict(conn, reattempt_date, train_flag)
    query = """select ml_model_id,
       version,
       start_timestamp,
       end_timestamp,
       ml_model_type_id,
       train_flag,
       trained_timestamp,
       avg_rmse,
       ml_rmse,
       feature_trained_start_timestamp,
       feature_trained_end_timestamp,
       model_name,
       currently_relevant,
       last_train_attempt_timestamp,
       observed_stop_count,
       median,
       average,
       model_blob
from ml_model
where current_timestamp between start_timestamp and end_timestamp
                         and train_flag = %s
                         and (last_train_attempt_timestamp is null or last_train_attempt_timestamp < %s)"""

    cursor = conn.cursor()
    log.info(f"Issuing query `{query}` with args {train_flag}, '{reattempt_date}'")
    cursor.execute(query, [train_flag, reattempt_date])
    results: [MLModel] = []
    row = cursor.fetchone()
    while row is not None:
        model_blob = None
        if row[17] is not None:
            model_blob = bytes(row[17])
        ml_model = MLModel(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8],
                           row[9], row[10], row[11], row[12], row[13], row[14], row[15], row[16], model_blob)
        if ml_model.ml_model_id in model_stop_dict:
            ml_model.model_stops = model_stop_dict[ml_model.ml_model_id]
        results.append(ml_model)
        row = cursor.fetchone()
    cursor.close()
    return results


def update_model_record(conn, model_record: MLModel):
    """record updated fields from model_record."""
    update_statement = """update ml_model set train_flag = %s,
        trained_timestamp = %s,
        avg_rmse = %s,
        ml_rmse = %s,
        feature_trained_start_timestamp = %s,
        feature_trained_end_timestamp = %s,
        last_train_attempt_timestamp = %s,
        observed_stop_count = %s,
        median = %s,
        average = %s,
        model_blob = %s
        where ml_model_id = %s"""
    cursor = conn.cursor()
    log.info(f"Issuing update_statement `{update_statement}` with args {model_record.train_flag}, "
             f"{model_record.trained_timestamp}, {model_record.avg_rmse}, {model_record.ml_rmse}, "
             f"{model_record.feature_trained_start_timestamp}, {model_record.feature_trained_end_timestamp} ,"
             f"<blob data>, {model_record.ml_model_id}")
    cursor.execute(update_statement, [model_record.train_flag,
                                      model_record.trained_timestamp,
                                      model_record.avg_rmse,
                                      model_record.ml_rmse,
                                      model_record.feature_trained_start_timestamp,
                                      model_record.feature_trained_end_timestamp,
                                      model_record.last_train_attempt_timestamp,
                                      model_record.observed_stop_count,
                                      model_record.median,
                                      model_record.average,
                                      model_record.model_blob,
                                      model_record.ml_model_id])
    conn.commit()
    cursor.close()
