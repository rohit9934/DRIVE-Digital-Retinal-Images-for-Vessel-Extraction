# DRIVE-Digital-Retinal-Images-for-Vessel-Extraction
We have done segmentation of blood vessels from their respective retinal images. As this is a segmentation model, we have used U-net architecture for the segmentation purpose. This is a binary classification task as at each pixel, we have to classify whether that pixel is feature or not.

# Dataset
 We have used [DRIVE](https://www.isi.uu.nl/Research/Databases/DRIVE/) dataset for training and testing purpose, this dataset consists of 40 images, 20 for training purpose and 20 for testing. 
# Working
* We are using DenseBlock U-net, the architecture is same as U-net but only difference is we are replacing 1 convolutional layer with Dense layer.
## perception
**base** : store template class of data_loader trainer infer model

**model** : store the definition of semantic segmention model

**infer** : how to predict

**trainer** : the way to train your model
# Get Started
1. First Clone the github repository.
* **For Training Purposes** First, you have to set all the paths in the files wherever necessary, in config directory and then run main_train.py
* **For Testing Purpose** You have to run the main_test.py file, All the results will be stored in experiments/VesselNet/checkpoint folder.
2. To modify the model, you can first change the path in all the python files and then change segmentionmodel.py  or to add the dense layer change thr denseunet.py file.

# Our Results
![ROC curve](https://github.com/rohit9934/DRIVE-Digital-Retinal-Images-for-Vessel-Extraction/tree/master/DRIVE/experiments/VesselNet/checkpoint/DRIVE_ROC.png)
 
![Models Comparsion](https://github.com/rohit9934/DRIVE-Digital-Retinal-Images-for-Vessel-Extraction/blob/master/DRIVE/image%20(1).png)

