import random

from models.DataProcessorForItems import DataProcessorForItems
from models.DataProcessorForUser import DataProcessorForUser
from models.ItemToItemApproach import ItemToItemApproach
from models.UserToUserApproach import UserToUserApproach


class CombinedApproach:

    """
    Combines user_to_user approach with item_to_item approach.
    Counts predictions for user_to_user approach
    Counts predictions for item_to_item approach
    Finding intersections between predictions for each user
    If intersection is less than 10, randomly adding items from predictions union.
    """

    def __init__(self, data, metrics, n_neighbours, item_asset, item_price, item_subclass, user_region, user_age):
        self.item_to_item = ItemToItemApproach(
            data,
            metrics,
            n_neighbours,
            DataProcessorForItems(item_asset, item_price, item_subclass))
        self.user_to_user = UserToUserApproach(
            data,
            metrics,
            n_neighbours,
            DataProcessorForUser(user_region, user_age))

    def get_multiple_prediction(self):

        items_answer = self.item_to_item.predictions_counter()
        users_answer = self.user_to_user.predictions_counter()
        combined_answer = dict()

        for k, v in items_answer.items():
            updated_set = set.intersection(set(v), set(users_answer[k]))
            if len(updated_set) < 10:
                while not len(updated_set) == 10:
                    updated_set.add(random.choice(v + users_answer[k]))
            combined_answer[k] = list(updated_set)
        return combined_answer

    def predictions_counter(self):
        return self.get_multiple_prediction()
