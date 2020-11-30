import ml_metrics as metrics


class SolutionAnalysis:

    """Counts MAP@10 score for model predictions"""

    def __init__(self, prediction, test_dataframe):
        self.prediction = prediction
        self.test = test_dataframe

    def count_map_at_10(self):
        """Counts mapk10 from ml_metrics"""
        self.test_reconfiguration()
        return metrics.mapk(self.prediction, self.test, 10)

    def test_reconfiguration(self):

        """
        Reconfigure self.test dataset intp valid form vor score counting.
        :return:
        """

        test_results = []
        prediction_results = []
        for each in self.test['row'].unique():
            test_interactions = list(self.test.loc[self.test['row'] == each]['col'].values)
            if len(test_interactions) > 0 and each in self.prediction.keys():
                test_results.append(test_interactions)
                prediction_results.append(self.prediction[each])
        self.test = test_results
        self.prediction = prediction_results
