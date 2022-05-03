# Covid_19_detection_from_Xray
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/13PSehBZsYRmbnY2-mKv0vnkEWxBcRSML?usp=sharing)
## Overview
COVID-19 is an infectious disease which has led to a dramatic loss of human life worldwide, its early detection is critical to control cases’ spreading and mortality. The actual leader diagnosis test is the Reverse transcription Polymerase chain reaction (RT-PCR)which is based on Nasopharyngeal swabs (NPSs).
<br>
<br>
This paper reports our experience with detection of COVID-19 using chest X-ray images(CXR). We here try to implement Un-Supervised, Supervised and Deep learning methods for binary image(COVID or NON-COVID) classification on CXR images. Implementation of PCA followed by application of K-means for the unsupervised techniques. In supervised learning we explored models such as Random Forest classifier, Decision Tree classifier, GaussianBayes. For the Deep Learning approach we developed a custom pipeline which involves usage of segmentation, SOTA(State of the art)CNN models and Multi-modular Approach.

## Dataset Links
[X-ray images](https://data.mendeley.com/datasets/8h65ywd2jr/3)
<br>
[Lung segmentation dataset](https://www.kaggle.com/code/nikhilpandey360/lung-segmentation-from-chest-x-ray-dataset/data)

## Built using:
- [Scikit Learn: ](https://scikit-learn.org/stable/) used ML Library
- [PyTorch: ](https://pytorch.org/) used ML Libraries
- [Streamlit: ](https://streamlit.io/) Javscript framework used
- [Pandas: ](https://pandas.pydata.org/) Python data manipulation libraries
- [Seaborn: ](https://seaborn.pydata.org/) Data visualisation library
- [Matplotlib: ](https://matplotlib.org/stable/index.html) Plots and Data visualization
- [OpenCV: ](https://opencv.org/) Image Processing
## Pipeline:
### 1. [combined_model.ipynb](https://colab.research.google.com/drive/13PSehBZsYRmbnY2-mKv0vnkEWxBcRSML?usp=sharing)
This is the main file with all the preprocessing, EDA, various Machine learning and Deep Learning Models.
- Installing libraries and dependency
- Importing the dataset 
- Exploratory Data Analysis and Visualisation
- Data Preprocessing
   - Normalization
   - Resizing, Random Rotation, Random Flips
   - conversion to Tensor
   - Flattening of images
- Workflow 
- <img src = "https://user-images.githubusercontent.com/73459839/166416746-36e164fc-8df6-4d95-835b-e461db633d14.png" width="400" height="250">
- Unsupervised Learning Techniques
   - PCA
   - K-means clustering
   - Agglomerative clustering
- Supervised Learning Techniques
   - Decision Tree Classifier
   - Random Forest Classifier
   - Gaussian Bayes
- Deep Learning Techniques
   - ResNet-18
   - ResNet-50
   - MobileNet-v3
   - VGG-16
   - VGG-19
- Image Segmentation and U-Net
   - Workflow 
   - <img src = "https://user-images.githubusercontent.com/73459839/166417044-40214fcd-2284-4866-88d8-eead53742f78.png" width="750" height="400">
   - The model returns the following 4 things :- 
       - 1) Mask 
       - 2) inverse (complimentary) Mask
       - 3) Masked image (Mask multiplied with original CXR image)
       - 4) Inverse Masked image (Inverse Mask multiplied with original CXR image)
- Combined Multi-Modular Approach
   - Workflow
   - <img src = "https://user-images.githubusercontent.com/73459839/166417760-8054abd5-e7f8-4733-8468-71026f98ab35.jpg" height = "600">
 
- Comparitive Analysis
   - Various Evaluation metrices used
   - Plots for comparing performance
   - Performance Table
## How to run:
- Run the cells according to above mentioned pipeline

## Deployment
<img src = "https://user-images.githubusercontent.com/73459839/166416272-5c7cace4-8276-41f2-9ecf-04bb4aba2694.jpg" width="750" height="750">

## Team Members
| Name  | Roll No. |
| ------------- | ------------- |
| [Mukul Shingwani](https://github.com/Mukulshingwani) | B20AI023 |
| [Saurabh Modi](https://github.com/SaurabhModi26)  | B20EE035 |
| [Mitarth Arora](https://github.com/mitarth-arora)  | B20EE096 |
