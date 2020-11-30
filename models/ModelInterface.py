from abc import ABCMeta, abstractmethod
from sklearn.neighbors import NearestNeighbors


class ModelInterface:

    """Interface for model construction"""

    __metaclass__ = ABCMeta

    def __init__(self, data, metrics, n_neighbours, data_processor):
        self.data = data
        self.metrics = metrics
        self.n_neighbours = n_neighbours
        self.data_processor = data_processor
        self.users_idx = []
        self.model = NearestNeighbors(metric=self.metrics, algorithm='brute', n_neighbors=self.n_neighbours, n_jobs=-1)

    def model_fitter(self):

        """
        Fitting NearestNeighbors model.
        """

        prepare_data, users_idx = self.data_processor.process_data(data=self.data)
        self.model.fit(prepare_data)
        self.users_idx = users_idx
        return prepare_data

    def model_nearest_neighbours_getter(self):

        """
        Finds nearest neighbour for all items/users.
        :return: nearest neighbours
        """
        prepare_data = self.model_fitter()
        return self.model.kneighbors(prepare_data, n_neighbors=self.n_neighbours)

    @abstractmethod
    def predictions_counter(self):

        """
        Making predictions. Should be overwritten for each approach.
        :return:
        """
        return
