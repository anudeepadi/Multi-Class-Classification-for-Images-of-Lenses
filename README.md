# gsoc_task
# Common Task: Multi-Class Classification for Images of Lenses
Problem Statement
The aim of this task is to build a model for classifying images of lenses into three classes, strong lensing images with no substructure, subhalo substructure, and vortex substructure. The images have been normalized using min-max normalization, and the task is to use PyTorch or Keras to classify the images.

Dataset
The dataset consists of 1000 images of lenses, with 333 images in each class. The images are of size 128x128 and have been normalized using min-max normalization. The dataset can be found in the 'dataset.zip' file on Google Drive.

Approach
The approach used for this task is to use the ImageDataGenerator class in TensorFlow for loading the data and performing data augmentation. The ImageDataGenerator class allows us to perform on-the-fly data augmentation, which can help improve the performance of the model.

The dataset is loaded using the flow_from_directory method of the ImageDataGenerator class, which takes the path to the directory containing the images and automatically generates the labels based on the subdirectories in the directory. The flow_from_directory method also allows us to specify the batch size, target size, and other parameters for loading the data.

For the model, InceptionResNetV2 is used. It is a deep convolutional neural network architecture that was developed by Google. It is a state-of-the-art architecture that combines the Inception and ResNet architectures, and it has been shown to achieve high accuracy on image classification tasks.

Using InceptionResNetV2 for the multi-class classification of images of lenses is a great choice as it has a large number of parameters and can handle complex image features. However, it is also a computationally expensive model, and training it may require a lot of resources. It is important to ensure that the hardware and resources are available to train and test the model properly. Additionally, it is important to ensure that the model is fine-tuned and optimized for the specific task of classifying images of lenses.

Results
The model was trained for 10 epochs, and the best model was saved based on the validation AUC score. The final model had a validation AUC score of 0.9644, which indicates that the model is able to accurately classify the images into the three classes.

The performance of the model can be improved by using more advanced data augmentation techniques, such as random rotations, translations, and shearing. Additionally, the model architecture can be improved by using techniques such as residual connections, dropout, and batch normalization.

Technologies Used
The technologies used for this task are Python and TensorFlow. The code is written using the Keras API in TensorFlow, and the ImageDataGenerator class is used for loading the data and performing data augmentation.

Conclusion
In this task, we built a model for classifying images of lenses into three classes using the ImageDataGenerator class in TensorFlow and a Convolutional Neural Network (CNN) architecture. The model was able to achieve a validation AUC score of 0.9644, indicating that it is able to accurately classify the images. The performance of the model can be further improved by using more advanced data augmentation techniques and improving the model architecture.
