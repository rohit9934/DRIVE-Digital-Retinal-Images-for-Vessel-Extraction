
"""
Copyright (c) 2019. All rights reserved.
Created by Rohit Sharma
"""


class InferBase(object):
    """
    For errors in infer file
    """

    def __init__(self, config):
        self.config = config  #giving values

    def load_model(self, name):
        """
        error in loading model
        """
        raise NotImplementedError

    def predict(self, data):
        """
        error in prediction
        """
        raise NotImplementedError
