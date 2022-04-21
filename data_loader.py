import datetime
from datetime import timedelta
import logging as log

import psycopg2

import config
import model_records


def connect(cfg: config.Config):
    """simple wrapper to connect to database with config.Config"""
    conn = psycopg2.connect(host=cfg.db_host,
                            database=cfg.db_name,
                            user=cfg.db_user,
                            password=cfg.db_password)
    return conn


class ObservedStopTime:
    """Holds records from observed_stop_time table, each row is an instance of a vehicle passing between stops"""
    def __init__(self,
                 row):
        self.data_set_id = row[0]
        self.trip_id = row[1]
        self.observed_time = row[2]
        self.scheduled_seconds = row[3]
        self.stop_id = row[4]
        self.stop_distance = row[5]
        self.next_stop_id = row[6]
        self.next_stop_distance = row[7]
        self.travel_seconds = row[8]
        self.scheduled_time = row[9]

    def __str__(self) -> str:
        return f"OST observed_time: {self.observed_time}, trip_id: {self.trip_id}, stop_id: {self.stop_id}, " \
               f"next_stop_id: {self.next_stop_id}, travel_seconds: {self.travel_seconds}"


class TripDeviation:
    """Holds records from trip_deviation table, each row is a delay measured in seconds from schedule. Early is
    negative, late is positive"""
    def __init__(self, row):
        self.created_at = row[0]
        self.trip_progress = row[1]
        self.trip_id = row[2]
        self.delay = row[3]

    def __str__(self):
        return f"TripDeviation created_at:{self.created_at}, trip_id:{self.trip_id}, delay:{self.delay}, " \
               f"trip_progress:{self.trip_progress},"


class RecordLoader:
    """Retrieves data relevant for the same trip from two open cursors for observed_stop_time (ost) and
    trip_deviation tables while moving chronologically in time"""
    def __init__(self,
                 ost_cursor,
                 trip_deviation_cursor):
        self._ost_cursor = ost_cursor
        self._trip_deviation_cursor = trip_deviation_cursor
        self._loaded_trip_deviations = []
        self._loaded_ost = []

    def _next_ost(self):
        """Retrieve the next ObservedStopTime from cursor, or None if cursor has no more records"""
        row = self._ost_cursor.fetchone()
        if row is None:
            return None
        return ObservedStopTime(row)

    def _next_trip_deviation(self):
        """Retrieve the next ObservedStopTime from cursor, or None if cursor has no more records"""
        row = self._trip_deviation_cursor.fetchone()
        if row is None:
            return None
        return TripDeviation(row)

    def remove_trip_osts_prior_to(self, expire_before):
        """Load next OST, removing any found to be before expire_before, including instances with the same trip id
        within an hour after the expire_before. OSTs loaded from the cursor are stored in _loaded_ost for later use.
        This is done to ensure OSTs for trips that where included in a prior days data are not also included in the
        current days data"""
        trip_ids_to_remove = {}
        finish_date = expire_before + timedelta(minutes=60)
        last_date_seen = expire_before - timedelta(minutes=60)
        while last_date_seen < finish_date:
            ost = self._next_ost()
            # no more data, so return
            if ost is None:
                return
            last_date_seen = ost.observed_time
            if last_date_seen < expire_before:
                # OST's trip_id should not be included
                trip_ids_to_remove[ost.trip_id] = True
            elif ost.trip_id not in trip_ids_to_remove:
                # only retain the OST if its not flagged as excluded
                self._loaded_ost.append(ost)

    def _load_hour_of_ost(self):
        """Load just over an hours worth of ost records"""
        first_date = None
        an_hour = timedelta(minutes=60)
        for ost in self._loaded_ost:
            if first_date is None:
                first_date = ost.observed_time
                continue
            if ost.observed_time - first_date > an_hour:
                return
        ost = self._next_ost()
        while ost is not None:
            self._loaded_ost.append(ost)
            if first_date is None:
                first_date = ost.observed_time

            if ost.observed_time - first_date > an_hour:
                return
            ost = self._next_ost()

    def get_next_ost_set(self, model_stop_list: [model_records.MLModelStops]) -> [ObservedStopTime]:
        """Search up to an hour's worth of OST from the same trip id"""
        self._load_hour_of_ost()
        results = []
        new_loaded_ost = []
        first_ost = None
        stop_model_matching = False
        for ost in self._loaded_ost:
            if first_ost is None:
                first_ost = ost
                results.append(ost)
            else:
                if not stop_model_matching and first_ost.trip_id == ost.trip_id:
                    if is_next_stop_in_model_stop_list(results, ost, model_stop_list):
                        results.append(ost)
                    else:
                        new_loaded_ost.append(ost)
                        stop_model_matching = True
                else:
                    new_loaded_ost.append(ost)
        self._loaded_ost = new_loaded_ost
        return results

    def get_trip_deviations(self,
                            trip_id: str,
                            before_timestamp: datetime.datetime,
                            expire_before: datetime.datetime) -> [TripDeviation]:
        """retrieve all TripDeviations for trip_id prior to before_timestamp
            expire any trip_deviations older than expire_before"""
        results = []
        new_loaded_trip_deviations = []
        furthest_seen = expire_before
        # iterate over previously loaded trips, and remove expired items
        for td in self._loaded_trip_deviations:
            if td.created_at < expire_before:
                continue
            furthest_seen = td.created_at
            if td.trip_id == trip_id and td.created_at < before_timestamp:
                results.append(td)
            else:
                new_loaded_trip_deviations.append(td)

        # reassign loaded_trip_deviations to remove expired items
        self._loaded_trip_deviations = new_loaded_trip_deviations

        while furthest_seen < before_timestamp:
            td = self._next_trip_deviation()
            if td is None:
                # no more results
                return results
            if td.created_at < expire_before:
                continue
            furthest_seen = td.created_at
            if td.trip_id == trip_id:
                if td.created_at < before_timestamp:
                    results.append(td)
            else:
                self._loaded_trip_deviations.append(td)

        return results

    def close_cursors(self):
        """Close all cursors used"""
        self._ost_cursor.close()
        self._trip_deviation_cursor.close()


def find_stop_id_position_in_list(stop_id: str, model_stop_list: [model_records.MLModelStops]) -> int:
    """
    Returns the index of stop_id in model_stop_list
    :param stop_id: stop_id to search for
    :param model_stop_list: list of MLModelStops to search
    :return: index of the stop_id in the list or -1 if not found
    """
    pos = 0
    for model_stop in model_stop_list:
        if model_stop.stop_id == stop_id:
            return pos
        pos += 1
    return -1


def is_next_stop_in_model_stop_list(previous_osts: [ObservedStopTime],
                                    ost: ObservedStopTime,
                                    model_stop_list: [model_records.MLModelStops]) -> bool:
    """
    Returns True if ost is present in model_stop_list in the same order if added to the end of previous_ost
    :param previous_osts: ObservedStopTime list prior to ost
    :param ost: next ObservedStopTime
    :param model_stop_list: MLModelStops to verify order/location in
    :return: true if ost occurs after previous_osts according to the order of model_stop_list
    """
    ost_position_in_list = find_stop_id_position_in_list(ost.stop_id, model_stop_list)
    size_previous = len(previous_osts)
    if size_previous < 1:
        return ost_position_in_list > -1
    last_previous = previous_osts[size_previous - 1]
    last_previous_in_list = find_stop_id_position_in_list(last_previous.stop_id, model_stop_list)
    return ost_position_in_list > last_previous_in_list


def prepared_record_loader(source_conn,
                           start_date: datetime.datetime,
                           end_date: datetime.datetime,
                           model_stop_list: [model_records.MLModelStops]) -> RecordLoader:
    """Creates RecordLoader with parameters between start_date and end_date relevant for stops in model_stop_list"""
    # end date needs to extend some minutes into the following day to find stop series that go into the following day
    end_date_buffered = end_date + timedelta(minutes=60)
    # query must start early to catch previous passage and trip deviations leading up to an hour away
    trip_query_start_date = start_date - timedelta(minutes=60)
    stop_query_part = prepare_stop_query_part(model_stop_list)

    ost_cursor = source_conn.cursor()
    ost_query = """select distinct data_set_id, trip_id, observed_time, scheduled_seconds, stop_id,
       stop_distance, next_stop_id, next_stop_distance, travel_seconds, scheduled_time from observed_stop_time
    where created_at between %s and %s
    and {} order by observed_time""".format(stop_query_part)
    log.info(f"Issuing query `{ost_query}` with args {start_date}, {end_date}")
    ost_cursor.execute(ost_query, (start_date, end_date_buffered))
    log.info("Query returned")

    trip_deviation_cursor = source_conn.cursor()
    trip_deviation_query = """with data_sets_and_trip_ids as (select distinct ost.data_set_id, ost.trip_id 
    from observed_stop_time ost
    where ost.created_at between %s and %s
    and {} )
    select td.created_at, td.trip_progress, td.trip_id, td.delay from trip_deviation td, data_sets_and_trip_ids
    where td.created_at between %s and %s
    and td.data_set_id = data_sets_and_trip_ids.data_set_id and td.trip_id = data_sets_and_trip_ids.trip_id
    order by td.created_at""".format(stop_query_part)
    log.info(f"Issuing query `{trip_deviation_query}` with args {start_date}, {end_date_buffered}, "
             f"{trip_query_start_date}, {end_date_buffered}")
    trip_deviation_cursor.execute(trip_deviation_query,
                                  (start_date, end_date_buffered, trip_query_start_date, end_date_buffered))
    log.info("Query returned")
    return RecordLoader(ost_cursor, trip_deviation_cursor)


def prepare_stop_query_part(model_stop_list: [model_records.MLModelStops]) -> str:
    """Create stop query part for sql query with stops in model_stop_list"""
    query_part = []
    for model_stop in model_stop_list:
        query_part.append(f"(stop_id = '{model_stop.stop_id}' and next_stop_id = '{model_stop.next_stop_id}')")

    return "({})".format("\n or ".join(query_part))




