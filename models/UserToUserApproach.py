from models.ModelInterface import ModelInterface


class UserToUserApproach(ModelInterface):

    """
    Prediction model for User to User approach.
    Checking distance between users with provided metric.
    Making predictions for each user with predictions_counter function
    """

    def predictions_counter(self):

        """
        Counting neighbours and distances for each user.
        Adding neighbours products in recommendations.
        """
        distances, neighbours = self.model_nearest_neighbours_getter()
        prediction_dict = dict()
        for idx, user in enumerate(self.users_idx):
            predictions = []
            for each in neighbours[idx][1::]:
                predictions += list(self.data.loc[self.data['row'] == each]['col'].values)
                if len(predictions) >= 10:
                    break
            prediction_dict[user] = predictions
        return prediction_dict
