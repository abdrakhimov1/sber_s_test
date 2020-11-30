from models.CombinedApproach import CombinedApproach
from models.DataProcessorForItems import DataProcessorForItems
from models.DataProcessorForUser import DataProcessorForUser
from models.ItemToItemApproach import ItemToItemApproach
from models.SolutionAnalysis import SolutionAnalysis
from models.UserToUserApproach import UserToUserApproach


class CustomGridSearch:

    """
    Custom grid search. Counts scores for different parameters. Returns best params with highest score
    """

    def __init__(
            self,
            metrics_list,
            n_neighbours_list,
            data,
            test,
            item_asset,
            item_price,
            item_subclass,
            user_region,
            user_age):

        self.metrics_list = metrics_list
        self.n_neighbours_list = n_neighbours_list
        self.data = data
        self.test = test
        self.item_asset = item_asset
        self.item_price = item_price
        self.item_subclass = item_subclass
        self.user_region = user_region
        self.user_age = user_age
        self.max_score = 0
        self.grid_result = []

    def grid_search(self):
        print("Starting grid search")
        for metric in self.metrics_list:
            for n_neighbours in self.n_neighbours_list:
                print('Params: ' + metric + ' ' + str(n_neighbours))

                user_to_user = UserToUserApproach(
                    self.data,
                    metric,
                    n_neighbours,
                    DataProcessorForUser(self.user_region, self.user_age))

                item_to_item = ItemToItemApproach(
                    self.data,
                    metric,
                    n_neighbours,
                    DataProcessorForItems(self.item_asset, self.item_price, self.item_subclass))

                # combined_method = CombinedApproach(
                #     self.data,
                #     metric,
                #     n_neighbours,
                #     self.item_asset,
                #     self.item_price,
                #     self.item_subclass,
                #     self.user_region,
                #     self.user_age)

                user_analyzer = SolutionAnalysis(user_to_user.predictions_counter(), self.test)
                user_score = user_analyzer.count_map_at_10()

                item_analyzer = SolutionAnalysis(item_to_item.predictions_counter(), self.test)
                item_score = item_analyzer.count_map_at_10()

                # combined_analyzer = SolutionAnalysis(combined_method.predictions_counter(), self.test)
                # combined_score = combined_analyzer.count_map_at_10()

                print('user_score: ' + str(user_score))
                print('item_score: ' + str(item_score))
                # print('combined_score: ' + str(combined_score))

                if user_score > self.max_score:
                    self.grid_result = [metric, n_neighbours, 'user_to_user']
                    self.max_score = user_score

                if item_score > self.max_score:
                    self.grid_result = [metric, n_neighbours, 'item_to_item']
                    self.max_score = item_score

                # if combined_score > self.max_score:
                #     self.grid_result = [metric, n_neighbours, 'combined_model']
                #     self.max_score = combined_score

    def best(self):
        return self.max_score, self.grid_result
