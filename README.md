# Covid_19_detection_from_Xray
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/13PSehBZsYRmbnY2-mKv0vnkEWxBcRSML?usp=sharing)
## Overview
COVID-19 is an infectious disease which has led to a dramatic loss of human life worldwide, its early detection is critical to control cases’ spreading and mortality. The actual leader diagnosis test is the Reverse transcription Polymerase chain reaction (RT-PCR)which is based on Nasopharyngeal swabs (NPSs).
<br>
This paper reports our experience with detection of COVID-19 using chest X-ray images(CXR). We here try to implement \textbf{Un-Supervised, Supervised and Deep learning methods} for binary image(COVID or NON-COVID) classification on CXR images. Implementation of PCA followed by application of K-means for the unsupervised techniques. In supervised learning we explored models such as Random Forest classifier, Decision Tree classifier, GaussianBayes. For the Deep Learning approach we developed a custom pipeline which involves usage of segmentation, SOTA(State of the art)CNN models and Multi-modular Models.

## Dataset Links
[X-ray images](https://data.mendeley.com/datasets/8h65ywd2jr/3)
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
### 1. [course_project](https://colab.research.google.com/drive/13PSehBZsYRmbnY2-mKv0vnkEWxBcRSML?usp=sharing)
This is the main file with all the preprocessing, EDA, various Machine learning and Deep Learning Models.
- Installing libraries and dependency
- Importing the dataset - [Flight Price Prediction Dataset ](https://drive.google.com/drive/folders/1tHNt5vPyCyKRQIitvGmf48AI2tna5xSk) 
- Exploratory Data Analysis and Visualisation
- Data Preprocessing - Basic preprocessing and cleaning the dataset
- Extra regressor model to determine feature importance
- Dividing the dataset into train and test
- Applying Machine Learning models
- Applying Deep Learning models
## How to run:
- Run the cells according to above mentioned pipeline
- 
## Team Members
| Name  | Roll No. |
| ------------- | ------------- |
| [Mukul Shingwani](https://github.com/Mukulshingwani) | B20AI023 |
| [Saurabh Modi](https://github.com/SaurabhModi26)  | B20EE035 |
| [Mitarth Arora](https://github.com/mitarth-arora)  | B20EE096 |
