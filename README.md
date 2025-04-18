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
- **Counter**: This is used to provide a tally and provide a count of the number of images and labels
- **Random**: This is used to randomize the images selected within each class to equal 373
- **Numpy**: This package is used for numerical operations with arrays
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
- **Google Colab**


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


## Section 4: References

[1] J, Arun Pandian. “Data for: Identification of Plant Leaf Diseases Using a 9-Layer Deep Convolutional Neural Network.” Mendeley Data, 18 Apr, 2019. [Online]. Available: data.mendeley.com/datasets/tywbtsjrjv/1. [Accessed: April 4, 2025]

[2] “Plantvillage Dataset.”Deep Lake, 2 June. 2023. [Online]. Available: datasets.activeloop.ai/docs/ml/datasets/plantvillage-dataset/#:~:text=The%20PlantVillage%20dataset%20is%20created,datasets%20with%20different%20background%20conditions. [Accessed: April 1, 2025].

[3] “Plantvillage.” Plantvillage [Online]. Available: plantvillage.psu.edu/. [Accessed: April 4, 2025]

[4] "What is ResNet-50?" Roboflow. [Online]. Available: https://blog.roboflow.com/what-is-resnet-50/. [Accessed: April 2, 2025]

[5] “VGG-19 Network.” MathWorks. [Online]. Available: https://www.mathworks.com/help/deeplearning/ref/vgg19.html. [Accessed: April 2, 2025].
