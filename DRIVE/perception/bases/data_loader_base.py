
"""
Copyright (c) 2018. All rights reserved.
Created by Rohit Sharma, Abdul Mugeesh and Kanishk Nama
"""


class DataLoaderBase(object):
    """
    Data Loaders.
    """

    def __init__(self, config):
        self.config = config  # Updating config info

    def prepare_dataset(self):
        """
        Errors and exception
        """
        raise NotImplementedError

    def get_train_data(self):
        """
        if training data not available
        """
        raise NotImplementedError

    def get_val_data(self):
        """
        if validation data not available
        """
        raise NotImplementedError
