# Data Appendix

## PlantVillage Tomato Dataset

### Unit of Observation:
The unit of observation is a single image of a tomato plant, labeled with its corresponding disease or healthy status.The dataset includes a numeric identifier for the disease class (numeric_label), a human-readable label (class_label), and the image data itself (image), represented as a 256x256 RGB array.


### Variables:
- **numeric_label**: Interger
- **class_label**: String
- **image**: uint8


### Descriptive Statistics:

#### Summary Statistics for class_label:
- **Defintion**: A label that identifies the plant's disease status. Since all the plant types are tomato, the label only varies by disease. There are ten different diseases represented in the dataset, resulting in ten unique class labels. 
- **missing values**: None
- **method to final form**: The data was obtained from the PlantVillage dataset, created by researchers at Penn State. We downloaded the dataset using tensorflow_datasets, and then used Python to filter the dataset to include only images of tomato plants. This filtering process resulted in a subset containing ten unique class labels, each indicating that the plant is a tomato and specifying the disease shown in the image.

 #### Figure 1: 

![image](https://github.com/user-attachments/assets/d580463b-4b4c-4251-9eb1-09f5fd187156)

This graph shows the distribution of images in each class after balancing.


#### Summary Statistics for numeric_label:
- **Defintion**: A number assigned to each class, ranging from 28 to 37. Each number corresponds directly to a specific class label.
- **missing values**: None
- **method to final form**: The data was obtained from the PlantVillage dataset, created by researchers at Penn State. We downloaded the dataset using tensorflow_datasets, and then used Python to filter the dataset to include only images of tomato plants. This filtering process resulted in a subset containing ten unique numeric_label values, ranging from 28 to 37.


#### Summary Statistics for Image:
- **Defintion**:  Image of Tomato plant.
- **missing values**: None
- **method to final form**: The data was obtained from the PlantVillage dataset, created by researchers at Penn State. We downloaded the dataset using tensorflow_datasets, and then used Python to filter the dataset to include only images of tomato plants. Also, during the data cleaning process we balanced the dataset so that each class_label and numeric_label had a total of 700 images.


#### Figure 2: 
![plantimage](https://github.com/user-attachments/assets/ea921f8c-2177-4204-ae57-1f183be69efa)

This shows sample images of the different tomato leaf diseases 


