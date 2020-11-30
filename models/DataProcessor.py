from abc import ABCMeta, abstractmethod


class DataProcessor:

    """
    Interface class for DataProcessors
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def process_data(self, data):
        return
