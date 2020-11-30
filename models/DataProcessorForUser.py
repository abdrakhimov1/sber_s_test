from models.DataProcessor import DataProcessor
import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
import random


class DataProcessorForUser(DataProcessor):

    def __init__(self, user_region_data, user_age_data):
        self.user_region_data = user_region_data
        self.user_age_data = user_age_data

    @staticmethod
    def data_preparation(user_region_data):

        """
        Making user dictionary from user  region or age
        :param user_region_data:
        :return: dictionary with age or region information.
        """

        df = user_region_data.drop_duplicates(subset='row', keep="last")
        user_region_dict = pd.Series(df.col.values, index=df.row).to_dict()
        return user_region_dict

    def user_information_combiner(self, train_pivot_table, user_region_data, user_age_data):

        """
        Adding information about user region and age into users pivot table.
        :param train_pivot_table: users pivot table
        :param user_region_data: user region dataframe
        :param user_age_data: user age dataframe
        :return: numpy matrix
        """

        data_size = len(train_pivot_table.index)

        regions = np.zeros(data_size, dtype=np.uint8)
        ages = np.zeros(data_size, dtype=np.uint8)

        user_region_dict = self.data_preparation(user_region_data)
        user_age_dict = self.data_preparation(user_age_data)

        regions_list = user_region_data['col'].unique()
        ages_list = user_age_data['col'].unique()

        for idx, i in enumerate(train_pivot_table.index):

            if i in user_region_dict.keys():
                regions[idx] = user_region_dict[i] / 10
            else:
                regions[idx] = random.choice(regions_list)

            if i in user_age_dict.keys():
                ages[idx] = user_age_dict[i]
            else:
                ages[idx] = random.choice(ages_list)

        numpy_pivot_table_in_unit8_format = train_pivot_table.values.astype(np.uint8)

        for each in [regions, ages]:
            numpy_pivot_table_in_unit8_format = np.hstack(
                (numpy_pivot_table_in_unit8_format, each.reshape((data_size, 1))))

        return numpy_pivot_table_in_unit8_format

    def process_data(self, data):

        """Data converter from pandas dataframe to sparse matrix"""

        train_pivot_table = data.pivot(
            index='row',
            columns='col',
            values='data',
        ).fillna(0)

        numpy_pivot_table_in_unit8_format = self.user_information_combiner(train_pivot_table, self.user_region_data, self.user_age_data)
        csr_matrix_for_users = csr_matrix(numpy_pivot_table_in_unit8_format)
        return csr_matrix_for_users, train_pivot_table.index
