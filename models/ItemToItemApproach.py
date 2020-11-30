from collections import Counter
from models.ModelInterface import ModelInterface


class ItemToItemApproach(ModelInterface):

    """
    Makes predictions with item to item approach.
    Finds nearest items for each userItem.
    Counts appearance of item for each users item recommendation with distance weight.
    Recommends top10 items with most weights for user.
    """

    def make_prediction_for_user(self, prediction_dict):
        users_list = self.data['row'].unique()
        return_dict = dict()
        for each_user in users_list:
            counter = Counter()
            for each_item in self.data.loc[self.data['row'] == each_user]['col'].values:
                if each_item in prediction_dict.keys():
                    for each_predicted_item, item_distance in zip(prediction_dict[each_item][0],
                                                                  prediction_dict[each_item][1]):
                        counter[each_predicted_item] += 1 * (1 - item_distance)
            return_dict[each_user] = [x[0] for x in counter.most_common(10)]
        return return_dict

    def predictions_counter(self):
        distances, neighbours = self.model_nearest_neighbours_getter()
        prediction_dict = dict()
        for idx, item in enumerate(self.users_idx):
            prediction_dict[item] = (neighbours[idx], distances[idx])
        return self.make_prediction_for_user(prediction_dict)
