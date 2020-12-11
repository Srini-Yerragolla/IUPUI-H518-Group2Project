# -IUPUI-H518-GroupProject
INFO-H518 Deep Learning, Group 2 Project  Fall 2020 SOIC
Problem statement
INFO-H518-Deep Learning group project Bird-species Recognition Using Convolutional Neural Network (CNN)
Purpose
The purpose of the group project is to apply the deep neural network algorithm to detect the bird species.


Dataset
The Caltech-UCSD Birds-200-2011 (CUB-200-2011) dataset is downloaded from the Kaggle website [1].   The dataset is extension of CUB-200 dataset [2].  It contains 200 bird species with 11,788 images.   Each species is associated with a Wikipedia article and organized by scientific classification (order, family, genus, species).
Data Exploration
Prior to implementing models, we proceeded with understanding of the data and its quality.  We noticed that total number of images per species were not equally distributed. We investigated how the sizes and aspect ratio of images are distributed.  
We plotted histogram of the height and width distribution of the birds’ image (figure 1).  From the histogram, we notice that the sizes of the images were not uniform.

Data Augmentation
To counter this imbalance in the number of images, we needed either make more images of some species or sub-sample of certain images. Implemented data augmentation (rotate, zoom, crop, flip, etc.) to increase the number of images.

Models
	We implemented four convolutional neural networks to classify the bird images.  The models are CNN, VGG16, ResNet50, and EfficientNetB7.  All model was created using Keras API.  
