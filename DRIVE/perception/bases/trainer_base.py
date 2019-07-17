"""
Copyright (c) 2018. All rights reserved.
Created by Rohit Sharma
"""


class TrainerBase(object):
    """
    Exceptions for and during training.
    """

    def __init__(self, model, data, config):
        self.model = model  
        self.data = data
        self.config = config 

    def train(self):
        """
        for other exceptions in training.
        """
        raise NotImplementedError
