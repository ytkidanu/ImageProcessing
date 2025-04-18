# Image Processing TEAM SBT (group 11)

## Section 1: Software and Platform Used

### Software:

**Python**: This is the primary programming language used for this project.

**TensorFlow**: This software library was used to support the neural network models we build for data analysis.

**Google Colab**: The cloud-based platform used for running Jupyter notebooks, particularly for data cleaning.

**Rivanna**: This UVA based high performance system platform enabled us to have suffiecent memory to download our image data and conduct analysis on it to build our model. More information can be found here: https://www.rc.virginia.edu/userinfo/hpc/


### Add-on Packages:

- **Matplotlib**: This package is to create visualization of the tomato images and EDA plots.
- **Tensorflow**: This package is to used to provide high performance numerical computing for our image data
- **tensorflow_datasets**: This package is used to provide access to the dataset
- **counter**: This is used to provide a tally and provide a count of the number of images and labels
- **random**: This is used to randomize the images selected within each class to equal 373
- **Numpy**: This package is used for numerical operations with arrays
- **Pandas**: 
- **defaultdict**: This is used for to account for missing data in dictionaries
- **gdown**: This package is used to access files that are on Google Drive
- **train_test_split**: This is used for to divide our data into training and test set
- **StandardScaler**: This is used to standarized our data when training our models
- **LinearSVC**: This is used to build our SVM model for further evaluation
- **confustion_matrix**: This is used to create a confusion matrix to asses how accurate the SVM model is.
- **classification_report**: This is used to provide a summary of how the model did with classification of the images
- **accuracy_score**: This is used to calculate and determine how accurate the model is
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
  - hi! 
- SCRIPTS (main folder)
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

These identified tomato labels were then saved as classes into a variable named tomato_classes. Further, we also saved the indices associated with each tomato class into a variable named tomato_label_indices. It is important to make sure here that the labels indicies are in the form of an integer. We created a function that filters through the various tomato images in each class and documents a true or false depending on if that tomato class matches a tomato_label_indices value.
Subsequently, we created a dataset that contains only images and label pair that correspond to being a tomato plant leaf image.

In order to ensure that this dataset contain the right impages, we then used Matplotlib to visualize these images with their label names.

**Assigned the Correct Images and Labels to the Right Classs**:
Although we have the classes and indices, we don't have the actual class label number that correspond with the number from the plant village dataset (28 - 37). In order to to this, we had a for loop that went through each pair of tomato image and integer label in the dataset and placed them in a blank set named unique_classes. Within the pair in this set, they were then assigned with the correct class number (28 - 37) from the dataset. We converted that to a list to make it easier to read and printed out to ensure that the proper class labels were pulled.

We then created a variable named label_names and populated that with all the tomato labels that were pulled: Tomato___Bacterial_spot, Tomato___Early_blight, Tomato___healthy ,Tomato___Late_blight,Tomato___Leaf_Mold, Tomato___Septoria_leaf_spot, Tomato___Spider_mites Two-spotted_spider_mite, Tomato___Target_Spot, Tomato___Tomato_mosaic_virus, Tomato___Tomato_Yellow_Leaf_Curl_Virus. We then connected the unqiue numeric label and unique class name for the tomato images depending on which class the originally came from. It is important here to adjust the base index to 28 to ensure that the proper labels are assigned to the correct classes. We print all the unique label here to ensure that each tomato class had the proper numberic class number associated with it.

**Balancing the Classes**:

In order to balance the classes to ensure that they all have the same number of images for modeling

### Build CNN Model
**Access the Chicago Data Portal**:

### Build SVM Model
**Access the Chicago Data Portal**:

### Evaluation Metrics for Each Model
**Access the Chicago Data Portal**:


## Section 4: References

[1] J, Arun Pandian. “Data for: Identification of Plant Leaf Diseases Using a 9-Layer Deep Convolutional Neural Network.” Mendeley Data, 18 Apr, 2019. [Online]. Available: data.mendeley.com/datasets/tywbtsjrjv/1. [Accessed: April 4, 2025]

[2] “Plantvillage Dataset.”Deep Lake, 2 June. 2023. [Online]. Available: datasets.activeloop.ai/docs/ml/datasets/plantvillage-dataset/#:~:text=The%20PlantVillage%20dataset%20is%20created,datasets%20with%20different%20background%20conditions. [Accessed: April 1, 2025].

[3] “Plantvillage.” Plantvillage [Online]. Available: plantvillage.psu.edu/. [Accessed: April 4, 2025]

[4] "What is ResNet-50?" Roboflow. [Online]. Available: https://blog.roboflow.com/what-is-resnet-50/. [Accessed: April 2, 2025]

[5] “VGG-19 Network.” MathWorks. [Online]. Available: https://www.mathworks.com/help/deeplearning/ref/vgg19.html. [Accessed: April 2, 2025].
