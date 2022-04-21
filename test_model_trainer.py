from datetime import datetime, timedelta
from unittest import TestCase

import data_loader
import model_records
import model_trainer
import test_data_loader


class TestStopTransitions(TestCase):

    def test_stop_transition_names(self):
        model_stop = test_ml_model_stops(1, '13721', '13722')
        stop_transition = model_trainer.StopTransitions(model_stop)
        column_names = stop_transition.column_names()
        self.assertEqual(2, len(column_names))
        self.assertEqual("previous_transition_t13722", column_names[0])
        self.assertEqual("previous_transition_age_t13722", column_names[1])

    def test_stop_transition_new_ost_matches(self):
        model_stop = test_ml_model_stops(1, '13721', '13722')
        stop_transition = model_trainer.StopTransitions(model_stop)
        ost_row = [
            1,
            "trip_id",
            datetime.fromisoformat("2022-01-02 06:17:57-08:00"),
            200,
            '13722',
            1000.0,
            '13723',
            1500.0,
            200,
            3600 * 9,
        ]
        ost_no_match = data_loader.ObservedStopTime(ost_row)

        self.assertFalse(stop_transition.new_ost(ost_no_match),
                         "Expected ost for different stop to cause new_ost to return False")

        ost_row[4] = '13721'
        ost_row[6] = '13722'
        ost_match = data_loader.ObservedStopTime(ost_row)

        self.assertTrue(stop_transition.new_ost(ost_match),
                        "Expected ost for same stop to cause new_ost to return True")

    def test_stop_transition_previous_ost_retention(self):
        model_stop = test_ml_model_stops(1, '13721', '13722')
        stop_transition = model_trainer.StopTransitions(model_stop)
        ost_row = [
            1,
            "trip_id",
            datetime.fromisoformat("2022-01-02 06:00:00-08:00"),
            200,
            '13721',
            1000.0,
            '13722',
            1500.0,
            200,
            3600 * 9,
        ]

        ost1 = data_loader.ObservedStopTime(ost_row)
        ost_row[2] = datetime.fromisoformat("2022-01-02 06:30:00-08:00")
        ost2 = data_loader.ObservedStopTime(ost_row)
        ost_row[2] = datetime.fromisoformat("2022-01-02 07:00:01-08:00")
        ost3 = data_loader.ObservedStopTime(ost_row)

        stop_transition.new_ost(ost1)

        self.assertEqual(0, len(stop_transition._previous_osts),
                         "expected first ost added to StopTransition to hold no previous osts")
        self.assertEqual(ost1, stop_transition._ost,
                         "expected StopTransition to retain last matching ost")

        stop_transition.new_ost(ost2)
        self.assertEqual(1, len(stop_transition._previous_osts),
                         "expected previous ost to be retained after adding another within an hour")
        self.assertEqual([ost1], stop_transition._previous_osts,
                         "expected StopTransition to retain last matching ost")
        self.assertEqual(ost2, stop_transition._ost,
                         "expected StopTransition to retain last matching ost")

        stop_transition.new_ost(ost3)
        self.assertEqual(1, len(stop_transition._previous_osts),
                         "expected previous ost to be retained after adding another within an hour and first ost to " +
                         "be dropped")
        self.assertEqual([ost2], stop_transition._previous_osts,
                         "expected StopTransition to retain last matching ost")
        self.assertEqual(ost3, stop_transition._ost,
                         "expected StopTransition to retain last matching ost")

    def test_stop_transition_get_last_transition(self):
        model_stop = test_ml_model_stops(1, '13721', '13722')
        stop_transition = model_trainer.StopTransitions(model_stop)
        ost_row = [
            1,
            "trip_id",
            datetime.fromisoformat("2022-01-02 06:00:00-08:00"),
            201,
            '13721',
            1000.0,
            '13722',
            1500.0,
            200,
            3600 * 9,
        ]

        ost1 = data_loader.ObservedStopTime(ost_row)
        ost_row[2] = datetime.fromisoformat("2022-01-02 06:30:00-08:00")
        ost2 = data_loader.ObservedStopTime(ost_row)
        ost_row[2] = datetime.fromisoformat("2022-01-02 06:50:00-08:00")
        ost3 = data_loader.ObservedStopTime(ost_row)

        stop_transition.new_ost(ost1)
        results = stop_transition.get_last_transition(datetime.fromisoformat("2022-01-02 06:10:00-08:00"))

        self.assertEqual(7200, results.age_value,
                         "expected transition results to be 7200 when there are no previous osts")
        self.assertEqual(ost1.scheduled_seconds, results.value,
                         "expected transition results to be scheduled time when there are no previous osts")

        stop_transition.new_ost(ost2)
        timestamp = datetime.fromisoformat("2022-01-02 06:10:00-08:00")
        results = stop_transition.get_last_transition(timestamp)

        self.assertEqual(10 * 60, results.age_value,
                         "expected transition results to be the distance from previous ost and provided timestamp")
        self.assertEqual(ost1.travel_seconds, results.value,
                         "expected transition results to be previous ost travel_seconds")

        stop_transition.new_ost(ost3)
        timestamp = datetime.fromisoformat("2022-01-02 06:20:00-08:00")
        results = stop_transition.get_last_transition(timestamp)
        self.assertEqual(20 * 60, results.age_value,
                         "expected transition results to be the distance from previous ost and provided timestamp")
        self.assertEqual(ost1.travel_seconds, results.value,
                         "expected transition results to be previous ost travel_seconds")

        # move timestamp past 2nd ost
        timestamp = datetime.fromisoformat("2022-01-02 06:31:00-08:00")
        results = stop_transition.get_last_transition(timestamp)
        self.assertEqual(60, results.age_value,
                         "expected transition results to be the distance from previous ost and provided timestamp")
        self.assertEqual(ost2.travel_seconds, results.value,
                         "expected transition results to be previous ost travel_seconds")


class TestModelTrainers(TestCase):

    def test_train_models_for_stop_list2(self):
        test_record_loader = test_data_loader.get_test_record_loader("test_ost_trip_deviations.json")
        model_stop_list = [test_ml_model_stops(1, '13721', '13722'),
                           test_ml_model_stops(2, '13722', '13723'),
                           test_ml_model_stops(3, '13723', '13724')]
        untrained_stop_models = {}
        for ms in model_stop_list:
            untrained_stop_models[ms.model_name] = True
        results = model_trainer.train_models_for_stop_list(test_record_loader, model_stop_list,
                                                           untrained_stop_models, False)

        test_json = test_data_loader.get_test_json("test_ost_trip_deviations.json")

        # reload test_record_loader to retrieve each ost
        test_record_loader = test_data_loader.get_test_record_loader("test_ost_trip_deviations.json")
        # iterate over each ost set, and ensure trip_deviations are present for each of them
        osts = test_record_loader.get_next_ost_set(model_stop_list)
        trip_number = 0
        while osts is not None and len(osts) > 0:
            trip_number += 1
            # check all stops
            first_ost = osts[0]
            last_ost = osts[len(osts) - 1]
            self.check_deviations_present(results.all_stop_trainer,
                                          test_json['trip_deviations'],
                                          first_ost.observed_time - timedelta(hours=1),
                                          last_ost.observed_time,
                                          last_ost.trip_id,
                                          trip_number)

            # check each ost
            for ost in osts:
                mt = get_model_trainer(results.stop_trainers, ost)
                self.assertTrue(mt is not None,
                                f"Expected to find model trainer for stop {ost.stop_id} to {ost.next_stop_id}")
                self.check_deviations_present(mt,
                                              test_json['trip_deviations'],
                                              ost.observed_time - timedelta(hours=1),
                                              ost.observed_time,
                                              ost.trip_id,
                                              trip_number)

            osts = test_record_loader.get_next_ost_set(model_stop_list)

        # clean up tmp files
        for mt in results.get_all_model_trainers():
            mt.clean_up()

    def check_deviations_present(self,
                                 trainer: model_trainer.ModelTrainer,
                                 deviation_rows: [data_loader.TripDeviation],
                                 start_time: datetime,
                                 end_time: datetime,
                                 trip_id: str,
                                 row_id: int):
        deviations = get_deviations(deviation_rows, trip_id, start_time, end_time)

        # get the rows and delete the csv file
        df = trainer.get_df()
        rows = df.loc[df['trip_number'] == row_id]

        self.assertEqual(len(deviations), rows.count()[0], "Expected a row for every deviation")
        deviation_row = 0
        for idx, row in rows.iterrows():
            dev = deviations[deviation_row]
            deviation_row += 1
            self.assertEqual(dev.delay, row['delay'],
                             "Expected to find relevant deviation expected for deviation {}".format(dev))



def get_deviations(deviation_rows,
                   trip_id: str,
                   start_time: datetime,
                   end_time: datetime):
    results = []
    for trip_deviation in deviation_rows:
        time = datetime.fromisoformat(trip_deviation[0])
        if time >= end_time:
            return results
        if time >= start_time and trip_deviation[2] == trip_id:
            results.append(data_loader.TripDeviation(trip_deviation))
    return results


def get_model_trainer(model_trainers: [model_trainer.ModelTrainer], ost: data_loader.ObservedStopTime):
    for mt in model_trainers:
        if mt.is_applicable(ost):
            return mt
    return None


def test_ml_model_stops(sequence, stop_id, next_stop_id):
    return model_records.MLModelStops(0, 0, sequence, stop_id, next_stop_id)
