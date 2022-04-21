import os.path
from unittest import TestCase

import data_loader
import json
import datetime

import model_records


class TestPrepareQueryPart(TestCase):
    def test_prepare_stop_query_part(self):
        model_stops = [test_ml_model_stops(0, '1001', '1002')]
        result = data_loader.prepare_stop_query_part(model_stops)
        expected = """((stop_id = '1001' and next_stop_id = '1002'))"""
        self.assertEqual(expected, result)

        model_stops.append(test_ml_model_stops(0, '1002', '1003'))
        expected = """((stop_id = '1001' and next_stop_id = '1002')
 or (stop_id = '1002' and next_stop_id = '1003'))"""
        result = data_loader.prepare_stop_query_part(model_stops)
        self.assertEqual(expected, result)


class CursorFixture:
    def __init__(self, rows, datetime_column):
        self._rows = rows
        self._datetime_column = datetime_column
        self._index = -1
        self._size = len(rows)

    def fetchone(self):
        if self._index + 1 < self._size:
            self._index += 1
            columns = self._rows[self._index]
            datetime_value = columns[self._datetime_column]
            columns[self._datetime_column] = datetime.datetime.fromisoformat(datetime_value)
            return columns
        return None

    def close(self):
        pass


class TestOstRetrieval(TestCase):

    def test_get_next_ost_set(self):
        record_loader = get_test_record_loader("test_ost_trip_deviations.json")
        model_stop_list = [test_ml_model_stops(1, '13721', '13722'),
                           test_ml_model_stops(2, '13722', '13723'),
                           test_ml_model_stops(3, '13723', '13724')]

        class ExpectedResults:
            def __init__(self, trip_id, stop_ids):
                self.trip_id = trip_id
                self.stop_ids = stop_ids

        expected_results = [ExpectedResults("11252291", ["13721", "13722", "13723"]),
                            ExpectedResults("11252292", ["13721", "13722", "13723"]),
                            ExpectedResults("11252293", ["13721", "13722", "13723"]),
                            ExpectedResults("11252164", ["13721"]),
                            ExpectedResults("11252164", ["13721", "13722", "13723"])]

        actual_results = []
        ost = record_loader.get_next_ost_set(model_stop_list)
        while len(ost) > 0:
            actual_results.append(ost)
            ost = record_loader.get_next_ost_set(model_stop_list)

        self.assertEqual(len(expected_results), len(actual_results), "expected results to be the same length")
        for i, actual_osts in enumerate(actual_results):
            expected = expected_results[i]
            for stop_index, ost_result in enumerate(actual_osts):
                expected_stop_id = expected.stop_ids[stop_index]

                self.assertEqual(expected.trip_id, ost_result.trip_id,
                                 f"expected result index {i},{stop_index} to have trip_id {expected.trip_id}")
                self.assertEqual(expected_stop_id, ost_result.stop_id,
                                 "expected result index {i},{stop_index} to have stop_id {expected_stop_id}")

    def test_trip_deviation_retrieval(self):
        record_loader = get_test_record_loader("test_ost_trip_deviations.json")
        before_timestamp = datetime.datetime.fromisoformat("2022-01-02 00:07:10-08:00")
        expire_before = datetime.datetime.fromisoformat("2022-01-01 23:01:26-08:00")

        trip_deviations = record_loader.get_trip_deviations("11252291", before_timestamp, expire_before)
        self.assertEqual(27, len(trip_deviations), "Expected 27 trip_deviations for trip 11252291")
        for i, trip_deviation in enumerate(trip_deviations):
            self.assertTrue(before_timestamp > trip_deviation.created_at,
                            f"trip_deviation was after before_timestamp, {before_timestamp} {trip_deviation}")
            self.assertTrue(expire_before < trip_deviation.created_at,
                            f"trip_deviation was before expire_timestamp, {expire_before} < {trip_deviation}")

        before_timestamp = datetime.datetime.fromisoformat("2022-01-02 00:37:07-08:00")
        expire_before = datetime.datetime.fromisoformat("2022-01-01 23:32:28-08:00")

        trip_deviations = record_loader.get_trip_deviations("11252292", before_timestamp, expire_before)
        self.assertEqual(44, len(trip_deviations), "Expected 27 trip_deviations for trip 11252291")
        for i, trip_deviation in enumerate(trip_deviations):
            self.assertTrue(before_timestamp > trip_deviation.created_at,
                            f"trip_deviation was after before_timestamp, {before_timestamp} {trip_deviation}")
            self.assertTrue(expire_before < trip_deviation.created_at,
                            f"trip_deviation was before expire_timestamp, {expire_before} < {trip_deviation}")


def get_test_json(file_path):
    test_data_file = open(os.path.join("test_data", file_path), "r")
    test_json = json.load(test_data_file)
    test_data_file.close()
    return test_json


def get_test_record_loader(file_path):
    test_json = get_test_json(file_path)
    return data_loader.RecordLoader(CursorFixture(test_json['ost'], 2),
                                    CursorFixture(test_json['trip_deviations'], 0), )


def test_ml_model_stops(sequence, stop_id, next_stop_id):
    return model_records.MLModelStops(0, 0, sequence, stop_id, next_stop_id)
