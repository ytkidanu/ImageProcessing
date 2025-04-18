# Image Processing TEAM SBT (group 11)

## Section 1: Software and Platform Used

### Software:

**Python**: This is the primary programming language used for this project.

**TensorFlow**: This software library was used to support the neural network models we build for data analysis.

**Google Colab**: The cloud-based platform used for running Jupyter notebooks, particularly for data cleaning.

**Rivanna**: This UVA based high performance system platform enabled us to have suffiecent memory to download our image data and conduct analysis on it to build our model. More information can be found here: https://www.rc.virginia.edu/userinfo/hpc/


### Add-on Packages:

- **Matplotlib**: This package is to create visualization of the tomato images and EDA plots.
- **Tensorflow**: This package is to used to provide high performance numerical computing for our image data and perfom modeling
- **tensorflow_datasets**: This package is used to provide access to the dataset
- **Numpy**: This package is used for numerical operations with arrays
- **Pandas**: This packages is used for working with the data to make it clean and ready for use for further analysis.
- **Scikit-learn**: This is used to build our SVM model and create a confusion matrix to asses how accurate the SVM model is 
- **seaborn**: This package is used to create visualization of the evaluation of how the SVM model performed for classification

### Platform:
- **Mac** and **Windows**: These operating systems were used during development. 


## Section 2: Map of Documentation

**DS-4002-GROUP-PROJECTS**
- DATA (main folder)
  - Cleaning_Dataset_and_Pre_model_Visualizations.ipynb
  - Data_Appendix.md
  - Images_labels_npyform
  - Obtaining_Plant_village_data_works.ipynb
  - balancing_augmenting_graphing(1).ipynb
  - tomato_balanced1
- OUTPUT (main folder)
  - ResNet50_Loss_over_epochs.png
  - ResNet50_Matrix.png
  - SVM_matrix.png
  - VGG19_Loss_over_Epoch.png
- SCRIPTS (main folder)
  - DATA_BALANCED_CNN_and_SVM.ipynb
  - Model SVM.ipynb
  - Model SVM_518_seed.ipynb
- LICENSE.md
- README.md

## Section 3: Instructions for reproducing our results. 
All code for reproducing can be found in the Data folder

### Obtaining our Data
**Accessing the PlantVillage Dataset**:
Since this dataset is public, it will downloaded from tensorflow_datasets. After installing and importing this package, we downloaded the plant_village dataset. This dataset contains multiple plant images, ranging from healthy to diseases leaf images. Within this code, be sure to include the argumnets with_info =True and as_supervised=True to ensure that the images are inputed into a dictionary that will have a pair of (image, label) from the image data.

Next, we saved this data into a dataset named 'full_dataset', with the (images, label) being training data.

Following downloading the data, we used Matplotlib and Pandas to display the images to ensure that they had been downloaded properly.

### Cleaning the Data
**Only Retaining the Tomato Plant Images and Labels**:
After loading in the data, we then pulled metadata about the plant village dataset and stored it into 'info' in order to get an understanding of how to clean this dataset to suit our research question and goal. We saved all label names of the different plant image categories into a variable named label_names and printed this variable to see which labels were the tomato ones. The tomato labels are: 28: Tomato___Bacterial_spot, 29: Tomato___Early_blight, 30: Tomato___healthy ,31: Tomato___Late_blight, 32: Tomato___Leaf_Mold, 33: Tomato___Septoria_leaf_spot, 34: Tomato___Spider_mites Two-spotted_spider_mite, 35: Tomato___Target_Spot, 36: Tomato___Tomato_mosaic_virus, 37: Tomato___Tomato_Yellow_Leaf_Curl_Virus.

These identified tomato labels were then saved as classes into a variable named tomato_classes. Further, we also saved the indices associated with each tomato class into a variable named tomato_label_indices. It is important to make sure here that the labels indicies are in the form of an integer. We created a function that filters through the various tomato images in each class and documents a true or false depending on if that tomato class matches a tomato_label_indices value and saved into a variable named tomato_dataset.
Subsequently, we created a dataset that contains only images and label pair that correspond to being a tomato plant leaf image.

In order to ensure that this dataset contain the right impages, we then used Matplotlib to visualize these images with their label names.

**Assigned the Correct Images and Labels to the Right Classs**:
Although we have the classes and indices, we don't have the actual class label number that correspond with the number from the plant village dataset (28 - 37). In order to to this, we had a for loop that went through each pair of tomato image and integer label in the dataset and placed them in a blank set named unique_classes. Within the pair in this set, they were then assigned with the correct class number (28 - 37) from the dataset. We converted that to a list to make it easier to read and printed out to ensure that the proper class labels were pulled.

We then created a variable named label_names and populated that with all the tomato labels that were pulled: Tomato___Bacterial_spot, Tomato___Early_blight, Tomato___healthy ,Tomato___Late_blight,Tomato___Leaf_Mold, Tomato___Septoria_leaf_spot, Tomato___Spider_mites Two-spotted_spider_mite, Tomato___Target_Spot, Tomato___Tomato_mosaic_virus, Tomato___Tomato_Yellow_Leaf_Curl_Virus. We then connected the unqiue numeric label and unique class name for the tomato images depending on which class the originally came from. It is important here to adjust the base index to 28 to ensure that the proper labels are assigned to the correct classes. We print all the unique label here to ensure that each tomato class had the proper numberic class number associated with it.

**Balancing the Classes**:

In order to balance the classes to ensure that they all have the same number of images for modeling, a blank set named class_counts was created and Counter was used to irterate through the tomato_dataset in a for loop to tally how many images were in each label. Following this, a min_tomato variable was created to store that the minimum images count among the classes was 373 (from Tomato_Tomato_mosaic_virus). We created a blank dictionary named class_samples that will be populated by a for loop that is convering our tomato_dataset into Numpy and placing our image and label pairs into it.
An empty tomato_balanced list is then populated with a values from a for loop where each class will be randomly sampled so that random images are chosen so it matches the minimum number of 373 images in each class. Next, to prepare for training our models, the list of image and label pairs are randomized so that the training model does not have as much bias.

### Build CNN Model

In order to build the CNN model, all the images in the balanced dataset were first resized to 224x224 pixels. This is the common dimension for ResNet-50 model, VGG-19 models, and InceptionV3. The images were also preprocessed, shuffled and batched. This ensures that our data is properly optimized for building our models.

**ResNet-50 Model**:
The first step is to create empty list for our images and label: image_list and label_list. Then we used a for loop to convert and flatten our dataset into the image and label pairs they were in before, convert all of these images and labels into a Numpy array, and populate our image_list and label_list, respectively. These list were then transformed and saved into numpy arrays as images_np and labels_np. Next, this data was split into a 80% train and 20 temporary split. The 80% will be used to train the model while the 20% will be used to split for validation and test.
Following this, we created tensorflow datasets from our x train and y train, x val and y val, and x test and y test variables. These newly created dataset were then batched. An instance named datagen was utilzied to apply random transformations to augment our data as the model is trained, with this being standarized. We then shifted our old labels from 28 - 37 to 0-0 and applied these labels to our y train, y val, and y test. Next, we mapped these new labels on to the corresponindg tomato class labels in the tensor dataset. 

Following this, we then build our ResNet-50 model with 10 output classes and trained it with the tensorflow training dataset. A validation tensorflow dataset was also created from the mapped y_val.
We then ploted training and validation losses over various epochs to see how the model performs over time.

ModelCheckpoint and Earlystopping are also employed as callback to evaluate our model, ensure that the best model is chosen, and not overtrain the model. 

**VGG-19 Model**:
This model is loaded in and the model layer weights are frozen so that the layer paramerter do not change while the model is training, slowing it down. With the prior model setting up all the aspect we need to build our model, the VGG-19 model uses the same tensorflow training mapped dataset and validation tensorflow mapped dataset.

ModelCheckpoint and Earlystopping are also employed as callback to evaluate our model, ensure that the best model is chosen, and not overtrain the model. 

**InceptionV3 Model**:
This model is loaded in and the model layer weights here are also frozen so that the layer parameters do not change while the model is training. With the prior models setting up all the aspect we need to build our model, this model uses the same tensorflow training mapped dataset and validation tensorflow mapped dataset.

ModelCheckpoint and Earlystopping are also employed as callback to evaluate our model, ensure that the best model is chosen, and not overtrain the model. 

### Build SVM Model
To begin, we combined our previsouly seperate x_test and x_val into one set and do the same for the y_test and y_val values. Further, our x_train and x_test arrays are flattened. This makes it so that row represent an image whereas the column is a pixel from that image. Following this, the x_train and x_test flatten arrays are scaled and standarized to ensure this data has the appropriate requirements to run a SVM model. 
Principal component analysis (PCA) is employed in order to reduce the dimentsion of the image data to reduce the computational load. Next, we employed GridSearchCV to determine the parameters (C, gamma, and kernel) that would ensure we get the best fit of our data for the model. Using those values and the x_train and y_train that underwent PCA, we built our final model.

### Evaluation Metrics for Each Model
In order to evaluate and compare how each model did classifiying tomato images, confusion matrices with heatmaps were generated for each model to visually represent how y_test values did compared to y_true values. Addtionally, classficication reports are calculated to see how well the model did at predicting for each class. It provides information how often the model correctly classifified a class and how times it was able to document how many of the images for each class the model was able to correctly classify. Comparing how each models did for each class, with higher precision and recall values indicating a better fit model. Overall model accuracy scores will also be calulated and compared between the models to determine which one has the highest accuracy and is therefore a better model for tomato leaf image classifaction.


## Section 4: References

[1] J, Arun Pandian. “Data for: Identification of Plant Leaf Diseases Using a 9-Layer Deep Convolutional Neural Network.” Mendeley Data, 18 Apr, 2019. [Online]. Available: data.mendeley.com/datasets/tywbtsjrjv/1. [Accessed: April 4, 2025]

[2] “Plantvillage Dataset.”Deep Lake, 2 June. 2023. [Online]. Available: datasets.activeloop.ai/docs/ml/datasets/plantvillage-dataset/#:~:text=The%20PlantVillage%20dataset%20is%20created,datasets%20with%20different%20background%20conditions. [Accessed: April 1, 2025].

[3] “Plantvillage.” Plantvillage [Online]. Available: plantvillage.psu.edu/. [Accessed: April 4, 2025]

[4] "What is ResNet-50?" Roboflow. [Online]. Available: https://blog.roboflow.com/what-is-resnet-50/. [Accessed: April 2, 2025]

[5] “VGG-19 Network.” MathWorks. [Online]. Available: https://www.mathworks.com/help/deeplearning/ref/vgg19.html. [Accessed: April 2, 2025].
