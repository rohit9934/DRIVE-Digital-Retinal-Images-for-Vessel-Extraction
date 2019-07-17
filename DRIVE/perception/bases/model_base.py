
"""
Copyright (c) 2018. All rights reserved.
Created by Rohit Sharma
"""


class ModelBase(object):
    """
    For exceptions in loading models and other similar problems
    """

    def __init__(self, config):
        self.config = config  
        self.model = None  

    def save(self):
        """
        loading checkpoint and hdf5 path
        """
        if self.model is None:
            raise Exception("[Exception] You have to build the model first.")

        print("[INFO] Saving model...")
        json_string = self.model.to_json()
        open(self.config.hdf5_path+self.config.exp_name + '_architecture.json', 'w').write(json_string)
        print("[INFO] Model saved")

    def load(self, checkpoint_path):
        """
       if path not available
        """
        if self.model is None:
            raise Exception("[Exception] You have to build the model first.")

        print("[INFO] Loading model checkpoint ...\n")
        self.model.load_weights(self.config.hdf5_path+self.config.exp_name+ '_best_weights.h5')
        print("[INFO] Model loaded")

    def build_model(self):
        """
        other exceptions
        """
        raise NotImplementedError
