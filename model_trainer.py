import csv
import datetime
import logging as log
import math
from typing import Dict

import numpy as np
import pandas
import xgboost as xgb
import holidays
import os
import tempfile
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupShuffleSplit

import data_loader
import model_records

MAX_TRANSITION_AGE = 3600

OBSERVED_HOLIDAYS = ["New Year's Day",
                     'Martin Luther King Jr. Day',
                     'Memorial Day',
                     'Independence Day',
                     'Labor Day',
                     'Thanksgiving',
                     'Christmas Day']


class Transition:
    """Convenience class to return names and values from StopTransition"""

    def __init__(self, name: str,
                 value: int,
                 age_name: str,
                 age_value: int):
        self.name = name
        self.value = value
        self.age_name = age_name
        self.age_value = age_value


class StopTransitions:
    """Tracks movement between stops for each stop pair in ModelTrainer using model_records.ObservedStopTime
    and produces Transitions for model training columns representing the time it took the last vehicle to move
    between those two stops and the number of seconds ago the vehicle traveled between those stops"""

    def __init__(self, model_stop: model_records.MLModelStops):
        self._model_stop = model_stop
        self._previous_osts = []
        self._ost = None
        self._col1_name = f'previous_transition_t{model_stop.next_stop_id}'
        self._col2_name = f'previous_transition_age_t{model_stop.next_stop_id}'

    def new_ost(self, ost):
        """Update current model_records.ObservedStopTime and push last one into previous OST list"""
        if self._model_stop.stop_id != ost.stop_id or self._model_stop.next_stop_id != ost.next_stop_id:
            return False
        if self._ost is not None:
            self._previous_osts.append(self._ost)

        # only keep hours worth of previous osts
        hour_before = ost.observed_time - datetime.timedelta(hours=1)
        previous_osts = []
        for previous_ost in self._previous_osts:
            if previous_ost.observed_time > hour_before:
                previous_osts.append(previous_ost)
        self._previous_osts = previous_osts
        self._ost = ost
        return True

    def column_names(self):
        """Retrieve list of column names for this stop transition"""
        return [self._col1_name, self._col2_name]

    def get_last_transition(self, timestamp):
        """Retrieves column names and values in Transition class for previous stop transition"""
        result = Transition(
            name=self._col1_name,
            value=self._ost.scheduled_seconds,
            age_name=self._col2_name,
            age_value=MAX_TRANSITION_AGE * 2
        )
        last_ost = None
        for ost in self._previous_osts:
            if ost.observed_time < timestamp:
                last_ost = ost
        if last_ost is None:
            return result
        transition_age = timestamp - last_ost.observed_time

        if transition_age.seconds > MAX_TRANSITION_AGE:
            return result

        result.value = last_ost.travel_seconds
        result.age_value = transition_age.seconds
        return result


class ModelTrainer:
    """Trains a model with one or more stop transitions (as described in model_stops). Each target is the travel time
    totaled from model_records.ObservedStopTime (OST) travel_seconds.
    Each trip requires all OSTs to be present for MLModelStops in model_stops. No rows will be produced if any OSTs are
    missing.
    A temporary CSV file is used to store records until training is performed.
    Intended use is:
        1. For all data available perform these two steps with data in chronological order.
             a. Call ModelTrainer.new_osts with the next set of OSTs for the same trip (same vehicle)
             b. Call ModelTrainer.new_deviation with data_loader.TripDeviations for the trip.
        2. Call ModelTrainer.train_model
        3. Check ModelTrainer.is_trained attribute if the model was successfully trained
        4. Use attributes avg_rmse, ml_rmse, median, average and json from get_model_json() to save model for later use.
        5. Call clean_up() to clean up temp files.

    """

    def __init__(self, model_stops: [model_records.MLModelStops]):
        self._pd = None
        self._model = xgb.XGBRegressor(objective='reg:squarederror')
        self.is_trained = False
        self.model_name = model_records.ml_model_stops_name(model_stops)
        self._stop_transitions = []
        self._single_stop_pair = False

        if len(model_stops) == 1:
            ms = model_stops[0]
            self._single_stop_pair = True
            self._first_stop_id = ms.stop_id
            self._last_stop_id = ms.next_stop_id

        for model_stop in model_stops:
            stop_transition = StopTransitions(model_stop)
            self._stop_transitions.append(stop_transition)

        self._can_produce_rows = False
        self._last_ost = None
        self._holiday = False
        self._schedule_seconds = 0
        self._actual_travel_seconds = 0
        self._start_deviation_at = None
        self.avg_rmse = None
        self.ml_rmse = None
        self.ost_count = 0
        self.median = None
        self.average = None
        self._temp_file = tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False)
        self._csv = csv.writer(self._temp_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        self._csv.writerow(self.get_columns())
        self._df = None

    def get_columns(self):
        """returns list of column names to be used by this model"""
        columns = get_shared_column_names()
        stop_transition: StopTransitions
        for stop_transition in self._stop_transitions:
            columns.extend(stop_transition.column_names())
        return columns

    def is_applicable_for_single_stop_pair(self, ost: data_loader.ObservedStopTime):
        """Is the OST applicable for this model containing a single stop pair"""
        return self._single_stop_pair and self._first_stop_id == ost.stop_id and self._last_stop_id == ost.next_stop_id

    def new_osts(self, osts: [data_loader.ObservedStopTime]):
        """Next set of OSTs available, all for the same trip."""
        if len(osts) != len(self._stop_transitions):
            self._can_produce_rows = False
            return
        self.ost_count += 1
        self._can_produce_rows = True
        self._last_ost = osts[len(osts) - 1]
        self._start_deviation_at = osts[0].observed_time - datetime.timedelta(hours=1)
        self._holiday = is_holiday(self._last_ost.observed_time)
        scheduled_seconds = 0
        count_matching_osts = 0
        actual_travel_seconds = 0
        for ost in osts:
            scheduled_seconds += ost.scheduled_seconds
            actual_travel_seconds += ost.travel_seconds
            for st in self._stop_transitions:
                if st.new_ost(ost):
                    count_matching_osts += 1
        # only produce rows if the set of OSTs filled all StopTransitions
        self._can_produce_rows = len(self._stop_transitions) == count_matching_osts
        self._schedule_seconds = scheduled_seconds
        self._actual_travel_seconds = actual_travel_seconds

    def close_csv(self):
        """clean up csv file after record collection is complete"""
        if not self._temp_file.closed:
            self._temp_file.close()

    def get_df(self):
        """returns pandas DataFrame with data loaded from temp csv file"""
        self.close_csv()
        if self._df is None:
            self._df = pandas.read_csv(self._temp_file.name)
        return self._df

    def train_model(self):
        """With records collected train the model and gather basic statistics"""
        df = self.get_df()
        row_count = df.shape[0]
        average = df["actual_travel_seconds"].mean()
        if not math.isnan(average):
            self.average = average
        median = df["actual_travel_seconds"].median()
        if not math.isnan(median):
            self.median = median
        if row_count < 30000:
            log.info(f"not enough rows ({row_count}), not training {self.model_name}")
            return

        train_inds, test_inds = next(GroupShuffleSplit(test_size=.20, random_state=7).split(
            df, groups=df['trip_number']))

        train_data = df.iloc[train_inds]
        test_data = df.iloc[test_inds]

        y_train = train_data["actual_travel_seconds"]
        x_train = train_data.drop(columns=["trip_number", "actual_travel_seconds"])

        y_test = test_data["actual_travel_seconds"]
        x_test = test_data.drop(columns=["trip_number", "actual_travel_seconds"])

        self._model.fit(x_train, y_train)

        prediction_array = self._model.predict(x_test)

        sched_rmse = np.sqrt(mean_squared_error(y_test, x_test["scheduled_seconds"]))

        avg_diffs = np.subtract(y_test, self.average)

        avg_squared_diff = np.square(avg_diffs)
        self.avg_rmse = np.sqrt(avg_squared_diff.mean())
        self.ml_rmse = np.sqrt(mean_squared_error(y_test, prediction_array))
        log.info(f"model {self.model_name}, rows: {row_count}, trip_count: {self.ost_count}, sched_rmse: {sched_rmse}, "
                 f"avg_rmse: {self.avg_rmse}, ml_rmse: {self.ml_rmse}")

        self.is_trained = True

    def new_deviation(self, trip_deviation: data_loader.TripDeviation):
        """Add single row for model training for this data_loader.TripDeviation, if possible"""
        if not self._can_produce_rows:
            return
        at = trip_deviation.created_at

        # ignore trip deviations outside the range of time we are interested in
        if self._last_ost.observed_time <= at or self._start_deviation_at > at:
            return

        new_row = [self.ost_count,
                   at.month,
                   at.weekday(),
                   at.hour,
                   at.minute,
                   at.second,
                   int(self._holiday is True),
                   self._schedule_seconds,
                   self._last_ost.scheduled_time,
                   trip_deviation.delay,
                   self._last_ost.next_stop_distance - trip_deviation.trip_progress,
                   self._actual_travel_seconds
                   ]

        for st in self._stop_transitions:
            transition = st.get_last_transition(trip_deviation.created_at)
            new_row.append(transition.value)
            new_row.append(transition.age_value)
        self._csv.writerow(new_row)

    def get_model_json(self):
        """returns model as json for later use"""
        # have to use temporary file with current version of xgboost
        if not self.is_trained:
            return None
        tf = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._model.save_model(tf.name)
        read_file = open(tf.name, "r")
        lines = read_file.readlines()
        read_file.close()
        os.remove(tf.name)
        return '\n'.join(lines)

    def clean_up(self):
        """removes temp csv file if present"""
        if os.path.exists(self._temp_file.name):
            os.remove(self._temp_file.name)
        self._df = None


class ModelTrainers:
    """
    Collection of ModelTrainer objects for training segment with model_stop_list.
    One model will be trained for all of these model_stop lists
    For any single stop pair that is not trained as flagged in untrained_stop_models another single pair model will
    be trained for that pair at the same time as the multi-stop model

    ModelTrainers encapsulates the use of ModelTrainer.
    Intended use is simular to ModelTrainer, but clean up is done automatically:
        1. For all data available perform these two steps with data in chronological order.
             a. Call ModelTrainer.new_ost_list with the next set of OSTs for the same trip (same vehicle)
             b. Call ModelTrainer.new_trip_deviation with data_loader.TripDeviations for the trip.
        2. Call train_models
        3. Call get_all_model_trainers to retrieve all models for saving results
    """
    def __init__(self,
                 model_stop_list: [model_records.MLModelStops],
                 untrained_stop_models: Dict[str, bool]):
        self.all_stop_trainer = ModelTrainer(model_stop_list)
        self.stop_trainers = []
        if len(model_stop_list) <= 1:
            return
        for ms in model_stop_list:
            model_name = model_records.ml_model_stops_name([ms])
            # only train stop models that require training
            if model_name in untrained_stop_models:
                self.stop_trainers.append(ModelTrainer([ms]))

    def new_ost_list(self, ost_list: [data_loader.ObservedStopTime]):
        """Calls all models with OSTs in ost_list"""
        self.all_stop_trainer.new_osts(ost_list)
        for ost in ost_list:
            self._new_ost_for_stop(ost)

    def _new_ost_for_stop(self, ost):
        for mt in self.stop_trainers:
            if mt.is_applicable_for_single_stop_pair(ost):
                mt.new_osts([ost])
                return

    def new_trip_deviation(self, trip_deviation: data_loader.TripDeviation):
        """Calls all models with new_deviation from trip_deviation list"""
        self.all_stop_trainer.new_deviation(trip_deviation)
        for mt in self.stop_trainers:
            mt.new_deviation(trip_deviation)

    def train_models(self):
        """Trains all models and cleans up"""
        self.all_stop_trainer.train_model()
        self.all_stop_trainer.clean_up()
        for mt in self.stop_trainers:
            mt.train_model()
            mt.clean_up()

    def get_all_model_trainers(self) -> [ModelTrainer]:
        """"returns list of all ModelTrainers"""
        results: [ModelTrainer] = [self.all_stop_trainer]
        for t in self.stop_trainers:
            results.append(t)
        return results


class PendingModels:
    """
    Contains dictionary of all models by name and list of all models with multiple pairs.
    """
    def __init__(self):
        self.all_models_by_name: Dict[str, model_records.MLModel] = {}
        self.multi_pair_models: [model_records.MLModel] = []


def get_shared_column_names():
    """returns list of column names shared by all models"""
    return ['trip_number',
            'month',
            'weekDay',
            'hour',
            'minute',
            'second',
            'holiday',
            'scheduled_seconds',
            'scheduled_time',
            'delay',
            'distance_to_stop',
            'actual_travel_seconds']


def is_holiday(at):
    """returns boolean if the day is a holiday recognized by service
    Currently only knows about holidays recognized by TriMet
    """
    holiday = holidays.US().get(at)
    return holiday is not None and holiday in OBSERVED_HOLIDAYS


def train_models_for_stop_list(record_loader: data_loader.RecordLoader,
                               model_stop_list: [model_records.MLModelStops],
                               untrained_stop_models: Dict[str, bool],
                               train: bool = True) -> ModelTrainers:
    """
    Train models for model_stop_list

    :param record_loader: prepared RecordLoader for stops in model_stop_list
    :param model_stop_list: list of MLModelStops to be trained
    :param untrained_stop_models: dictionary of model names not yet trained
    :param train: true if the models should actually be trained. Set to False for testing trainer loading
    :return: ModelTrainers used to train stops in model_stop_list

    """
    model_trainers = ModelTrainers(model_stop_list, untrained_stop_models)
    ost_list = record_loader.get_next_ost_set(model_stop_list)
    while len(ost_list) > 0:
        first = ost_list[0]
        last = ost_list[len(ost_list) - 1]
        model_trainers.new_ost_list(ost_list)
        trip_deviations = record_loader.get_trip_deviations(last.trip_id,
                                                            last.observed_time,
                                                            first.observed_time - datetime.timedelta(hours=1))

        for td in trip_deviations:
            model_trainers.new_trip_deviation(td)
        ost_list = record_loader.get_next_ost_set(model_stop_list)
    if train:
        model_trainers.train_models()
    return model_trainers


def train_pending_models(conn, start_date: datetime, end_date: datetime, reattempt_date: datetime):
    """
    For all data between start_date and end_date train any relevant untrained model. Reattempt previously attempted
    models still needing training if reattempt_date is after reattempt_date
    """
    pending_models = get_pending_models(conn, reattempt_date)
    untrained_stop_models: Dict[str, bool] = {}

    # collect all stop pair models
    for stop_pair_name in pending_models.all_models_by_name.keys():
        untrained_stop_models[stop_pair_name] = True

    # first train all multi pair models
    for model in pending_models.multi_pair_models:
        train_models(conn, start_date, end_date, pending_models, untrained_stop_models, model)

    # then train all pending single stop pairs
    for model_name in untrained_stop_models.copy().keys():
        model = pending_models.all_models_by_name[model_name]
        train_models(conn, start_date, end_date, pending_models, untrained_stop_models, model)


def train_models(conn,
                 start_date: datetime,
                 end_date: datetime,
                 pending_models: PendingModels,
                 untrained_stop_models: Dict[str, bool],
                 model: model_records.MLModel):
    """prepares record_loader, trains model, and records results"""
    record_loader = data_loader.prepared_record_loader(conn, start_date, end_date, model.model_stops)
    model_trainers = train_models_for_stop_list(record_loader, model.model_stops, untrained_stop_models)
    record_model_trainers_results(conn, model_trainers.get_all_model_trainers(), pending_models,
                                  untrained_stop_models, start_date, end_date)


def record_model_trainers_results(conn,
                                  model_trainers: [ModelTrainer],
                                  pending_models: PendingModels,
                                  untrained_stop_models: Dict[str, bool],
                                  data_start_date: datetime.datetime,
                                  data_end_date: datetime.datetime):
    """records training results for all models in model_trainers to database,
    removes entries in untrained_stop_models if present"""
    for mt in model_trainers:
        if mt.model_name not in pending_models.all_models_by_name:
            log.warning(f"Missing MLModel record for {mt.model_name}, unable to save results")
            continue
        model_record = pending_models.all_models_by_name[mt.model_name]
        record_model_results(conn, model_record, mt, data_start_date, data_end_date)
        if mt.model_name in untrained_stop_models:
            untrained_stop_models.pop(mt.model_name)


def record_model_results(conn,
                         model_record: model_records.MLModel,
                         model_trainer: ModelTrainer,
                         data_start_date: datetime.datetime,
                         data_end_date: datetime.datetime):
    """Save results from model_trainer to model_record"""
    now = datetime.datetime.now()
    if model_trainer.is_trained:
        model_record.train_flag = False
        model_record.model_blob = bytes(model_trainer.get_model_json(), 'us-ascii')
        model_record.avg_rmse = model_trainer.avg_rmse
        model_record.ml_rmse = model_trainer.ml_rmse
        model_record.feature_trained_start_timestamp = data_start_date
        model_record.feature_trained_end_timestamp = data_end_date
        model_record.trained_timestamp = now

    model_record.last_train_attempt_timestamp = now
    model_record.observed_stop_count = model_trainer.ost_count
    model_record.median = model_trainer.median
    model_record.average = model_trainer.average
    model_records.update_model_record(conn, model_record)


def get_pending_models(conn, reattempt_date: datetime) -> PendingModels:
    """
    Retrieves all models currently relevant and needing training, or needing to be reattempted after reattempt_date
    """
    pending_models = PendingModels()
    current_models = model_records.get_current_models(conn, reattempt_date, True)
    for m in current_models:
        if len(m.model_stops) > 1:
            pending_models.multi_pair_models.append(m)
        pending_models.all_models_by_name[m.model_name] = m
    return pending_models
