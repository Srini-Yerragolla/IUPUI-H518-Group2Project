# IUPUI-H518-Group2 Deep Learning Project
INFO-H518 Deep Learning, Group 2 Project  Fall 2020 SOIC
Problem statement
INFO-H518-Deep Learning group project Bird-species Recognition Using Convolutional Neural Network (CNN)

Introduction
There are numerous species of birds on the planet earth and birds hold a significant place in various cultures. Bald Eagle is a symbol of strength and stamina and is a national symbol of the United States. Some birds are found in our neighborhood chirping and coexisting with us in the city. A bird enthusiast can recognize the bird species with ease. But for machines to classify the images, it is a tough job. 
For our research, we applied a deep learning framework to identify the species of birds. Our research focused on identifying various algorithms that will fit the dataset accurately. We leveraged the bird’s image dataset that was publicly available and pre-trained algorithm for better categorization. 

Dataset
We used a well-known Caltech- UCSD bird is 200-2011 (CUB-200-2011) dataset with images of various birds’ species [1]. The dataset is an extension of the CUB-200 dataset [2] [3].  It contains 200 bird species with 11,788 images.   Each species is associated with a Wikipedia article and organized by scientific classification (order, family, genus, and species). The dataset has annotations such that includes 312 binary attributes, 15 component positions, 1 bounding box.

Methodology
Our methodology is outlined in the block diagram below [4]. The method consists of four processes 1. Data Input/ Loading images 2. Data augmentation 3. Model implementation 4. Prediction and 5. Evaluation of the model (figure 1).  Each step is described in further detail in the sections

Data Exploration
Before implementing models, we proceeded with understanding of the data and its quality.  We noticed that the total number of images per species were not equally distributed. We investigated how the sizes and aspect ratio of images are distributed.  
We plotted a histogram of the height and width distribution of the birds’ image (figure 2).  From the histogram, we notice that the sizes of the images were not uniform.

Data Augmentation
To counter this imbalance in the number of images, we needed to either make more images of some species or sub-sample of certain images. Implemented data augmentation (rotate, zoom, crop, flip, etc.) to increase the number of images.

We used Keras to create new augmented images. Figure 3 shows the output of newly generated new images in an augmented folder.
We perform data augmentation to train the data very well to predict accurately. The above picture shows the Distribution of width and height across the CUB-200-2011 (birds) dataset.

Models Implemented
	We used four convolutional neural networks to classify the bird images.  The models were CNN, VGG16, ResNet50, and EfficientNetB7.  All models were implemented using the Keras API [5] [6].  
Convolutional Neural Network (CNN)
CNN is a widely used artificial neural network in image analysis and compared to traditional MLP, CNN decreases the computational scale.  It also helps to extract important features in artificial intelligence. 
Evaluation and concerns: The accuracy of recognizing reflects the performance of the proposed algorithm. We applied CNN, ResNet50, EfficientNetB7, VGG16 models. We also tried Dens net but after our training and test accuracy, the data did not fit accurately in this model.  We were concerned that the training data may not be enough since there are only around 60 examples for each species of bird. So, we did data augmentation.  We split the data into training, validation and testing sets [7]. 

Result
CNN model was trained on 100 epochs on the Adam optimizer and has a training accuracy of 78%, validation accuracy of 88%. See figure 5 for the CNN model result.

VGG 16
VGG16 is a convolutional neural network model proposed by K. Simonyan and A. Zisserman from the University of Oxford in the paper “Very Deep Convolutional Networks for Large-Scale Image Recognition”. The VGG-16 is a simpler architecture model since it does not use many hyperparameters. It always uses 3 x 3 filters with stride of 1 in the convolution layer and uses same padding in pooling layers 2 x 2 with stride of 2 [8].

Result
VGG 16 model was trained on 100 epochs on the RMSprop optimizer with the learning rate of 0.0001 has training accuracy of 79%, validation accuracy of 48%. See figure 7 for VGG16 training and validation output.

Resnet50
Residual learning framework is the easiest to train networks. ResNet was the winner of ILSRVC 2015, also called Residual Neural Network (ResNet) by Kaiming. This architecture introduced a concept called “skip connections''. Typically, the input matrix calculates in two linear transformations with the ReLU activation function. In the Residual network, it directly copies the input matrix to the second transformation output and sums the output in final the ReLU function. 
ResNet50 is a variant of the ResNet model which has 48 Convolution layers along with one Maximum Pooling and one Average Pooling layer. It has 3.8 x 10^9 Floating points operations. It is a widely used ResNet model and we have explored ResNet50 architecture in depth [9]. 

Result
ResNet50 model was trained on 100 epochs on the Adam optimizer with the learning rate of 0.0001 has a training accuracy of 92% and validation accuracy of 62%. See figure 8 for the result.
EfficientNetB7
The EfficientNetB7 model was proposed by Mingxing Tan and Quoc V. Le of Google Research, Brain team in their research paper ‘EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks’. Based on the EfficientNet- Bo architecture, thee engineers at google brain developed this model that optimizes both accuracy and floating-point operations [10]. 

Efficient Nets, as the name suggests are much efficient computationally and also achieved good results. 

Result	
EfficientNetB7 model was trained on 50 epochs on Adam optimizer with the learning rate of 0.01 has training accuracy 73% and validation accuracy 61%. Figure 9 shows the model training and validation accuracy.

Conclusions
We applied the Convolution neural network (CNN), VGG16, ResNet50, and EfficieNetB7 models successfully. The VGG16 and ResNet50 models gave us a very good accuracy rate. We learned that a vast number of image processing through convolution neural networks takes a lot of time to process and it needs a resource-intensive operation. We noticed that CPU runtime took a long time and stopped several times. As a result, we switched to the GPU runtime. The number of epochs plays a significant role to improve the accuracy of the model. We also tried other models such as DenseNet, but the dataset did not fit the model.

Contributions 
*These authors contributed equally to this work Amit Parulekar: aparulek@iu.edu,  Pari Brown: pkavoosi@iu.edu, Srini Yerragolla: syerrago@iu.edu.


	
