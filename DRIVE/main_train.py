
"""

Created by Team 34 during 2019 summer internship at Leadingindia.ai Bennett University.
All rights reserved: Rohit Sharma, Abdul Mugeesh, Kanishk Nama.


"""
import os
import sys
sys.path.insert(0,"/home/dgxuser102/data/team34/experiments")
#sys.path.insert(0,"/home/dgxuser102/team34/experiments")
#sys.path.insert(0,"/home/dgxuser102/team34/experiments/experiments/data_loaders")
#sys.path.insert(0,"/home/dgxuser102/data/team34/experiments/experiments/data_loaders")
sys.path.insert(0,"/home/dgxuser102/data/team34/experiments/experiments/data_loaders/standard_loader.py")
sys.path.insert(0,"/home/dgxuser102/data/team34/experiments/configs/utils/config_utils.py")
sys.path.insert(0,"/home/dgxuser102/data/team34/experiments/configs/utils/img_utils.py")

sys.path.insert(0,"/home/dgxuser102/data/team34/experiments/configs/segmention_config.json")
sys.path.insert(0,"/home/dgxuser102/data/team34/experiments/perception/trainers/segmention_trainer.py")
sys.path.insert(0,"/home/dgxuser102/data/team34/experiments/perception/models/segmention_model.py")
sys.path.insert(0,"/home/dgxuser102/data/team34/experiments/perception/models/dense_unet.py")
sys.path.insert(0,"/home/dgxuser102/data/team34/experiments/perception/metric/segmention_metric.py")
sys.path.insert(0,"/home/dgxuser102/data/team34/experiments/perception/infers/segmention_infer.py")
sys.path.insert(0,"/home/dgxuser102/data/team34/experiments/perception/bases/")
from experiments.data_loaders.standard_loader import DataLoader
from perception.models.dense_unet import  SegmentionModel
from perception.trainers.segmention_trainer import SegmentionTrainer
from configs.utils.config_utils import process_config
import numpy as np


sys.path.insert(0,'/home/dgxuser102/team34/experiments/Graphviz2.38/bin/')
sys.path.insert(0,'/home/dgxuser102/data/team34/experiments/Graphviz2.38/bin/')
sys.path.insert(0,'/usr/local/lib/python3.7/dist-packages/graphviz-0.11.1.dist-info/')

os.environ["PATH"] += os.pathsep + '/usr/local/lib/python3.7/dist-packages/graphviz/'

def main_train():
    print('[INFO] Reading configuration files')

    config = None

    try:
        config = process_config('/home/dgxuser102/data/team34/experiments/configs/segmention_config.json')
    except Exception as e:
        print('[Exception] Configuration Error, %s' % e)
        exit(0)
    # np.random.seed(47) 
    print('[INFO] Preparing Data...')
    dataloader = DataLoader(config=config)
    dataloader.prepare_dataset()

    train_imgs,train_gt=dataloader.get_train_data()
    val_imgs,val_gt=dataloader.get_val_data()

    print('[INFO] Using our model to train...')
    model = SegmentionModel(config=config)
    #
    print('[INFO] Now Training...')
    trainer = SegmentionTrainer(
         model=model.model,
         data=[train_imgs,train_gt,val_imgs,val_gt],
         config=config)
    trainer.train()
    print('[INFO] Finishing the training...')



if __name__ == '__main__':
    main_train()
    # test_main()
