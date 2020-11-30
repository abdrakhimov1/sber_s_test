from scipy.sparse import csr_matrix

from models.DataProcessor import DataProcessor
import pandas as pd
import numpy as np


class DataProcessorForItemsWithoutUserInfo(DataProcessor):

    """
    Data processor for item-item approach without Users vectors.
    """

    def __init__(self, item_asset_data, item_price_data, item_subclass_data):
        self.item_asset_data = item_asset_data
        self.item_price_data = item_price_data
        self.item_subclass_data = item_subclass_data

    @staticmethod
    def data_preparation(data):

        """
        Converting information about item assets and prices into dictionary
        :param data:
        :return: dictionary of items
        """

        df = data.drop_duplicates(subset='row', keep="last")
        item_asset_dict = pd.Series(df.data.values, index=df.row).to_dict()
        return item_asset_dict

    @staticmethod
    def data_preparation_for_subclasses(data):

        """
        Converting information about item subclass into dictionary
        :param data:
        :return: dictionary of items
        """

        df = data.drop_duplicates(subset='row', keep="last")
        item_asset_dict = pd.Series(df.col.values, index=df.row).to_dict()
        return item_asset_dict

    def process_data(self, data):

        """
        Data converter from pandas dataframe to sparse matrix
        :param data:
        :return: csr_matrix with items
        """

        item_assets_dict = self.data_preparation(self.item_asset_data)
        item_prices_dict = self.data_preparation(self.item_price_data)
        item_subclasses_dict = self.data_preparation_for_subclasses(self.item_subclass_data)

        prepared_data = []
        items_idx = []
        for each in item_assets_dict.keys():
            if each in item_prices_dict.keys() and each in item_subclasses_dict.keys():
                prepared_data.append([item_assets_dict[each] * 100, item_prices_dict[each] * 100, item_subclasses_dict[each] / 1000])
                items_idx.append(each)

        numpy_table = np.array(prepared_data, dtype=float)
        csr_matrix_for_users = csr_matrix(numpy_table)
        print(numpy_table)
        return csr_matrix_for_users, items_idx
