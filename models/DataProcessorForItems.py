from models.DataProcessor import DataProcessor
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd


class DataProcessorForItems(DataProcessor):

    """
    Data processor for items.
    Combines all information about items in csr_matrix
    This version includes users vector(information about users, who interacted items)

    """

    def __init__(self, item_asset_data, item_price_data, item_subclass_data):
        self.item_asset_data = item_asset_data
        self.item_price_data = item_price_data
        self.item_subclass_data = item_subclass_data

    @staticmethod
    def data_preparation(data):

        """
        Collects all items in one dict
        :param data:
        :return: dictionary containing items
        """

        df = data.drop_duplicates(subset='row', keep="last")
        item_asset_dict = pd.Series(df.data.values, index=df.row).to_dict()
        return item_asset_dict

    def items_information_combiner(self, train_pivot_table, item_asset_data, item_price_data, item_subclass_data):

        """
        Adding asset, price and subclass information into item-users pivot table.
        :param train_pivot_table: ready pivot table with users as columns, items as rows with values [0, 1]
        :param item_asset_data: item_asset dataframe
        :param item_price_data: item_price dataframe
        :param item_subclass_data: item_subclass dataframe
        :return: numpy matrix
        """

        data_size = len(train_pivot_table.index)

        assets = np.zeros(data_size, dtype=np.uint8)
        prices = np.zeros(data_size, dtype=np.uint8)
        subclasses = np.zeros(data_size, dtype=np.uint8)

        item_assets_dict = self.data_preparation(item_asset_data)
        item_prices_dict = self.data_preparation(item_price_data)
        item_subclasses_dict = self.data_preparation(item_subclass_data)

        assets_mean = item_asset_data['data'].mean()
        prices_mean = item_price_data['data'].mean()
        subclasses_mean = item_subclass_data['data'].mean()

        for idx, i in enumerate(train_pivot_table.index):

            try:
                assets[idx] = item_assets_dict[i]
            except IndexError:
                assets[idx] = assets_mean

            try:
                prices[idx] = item_prices_dict[i]
            except IndexError:
                prices[idx] = prices_mean

            try:
                subclasses[idx] = item_subclasses_dict[i]
            except IndexError:
                subclasses[idx] = subclasses_mean

        numpy_pivot_table_in_unit8_format = train_pivot_table.values

        for each in [assets, prices, subclasses]:
            numpy_pivot_table_in_unit8_format = np.hstack(
                (numpy_pivot_table_in_unit8_format, each.reshape((data_size, 1))))

        return numpy_pivot_table_in_unit8_format

    def process_data(self, data):

        """
        Data converter from pandas dataframe to sparse matrix
        """

        train_pivot_table = data.pivot(
            index='col',
            columns='row',
            values='data'
        ).fillna(0)
        numpy_pivot_table_in_unit8_format = train_pivot_table.values.astype(np.uint8)
        csr_matrix_for_users = csr_matrix(numpy_pivot_table_in_unit8_format)
        return csr_matrix_for_users, train_pivot_table.index
