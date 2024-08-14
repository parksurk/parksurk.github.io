---
title: "UpStage AI LAB 2nd Competition - CV CLASSIFICATION"
excerpt: "This blog introduces the journey of the Digital Treasure Quest team as they participated in a document type classification competition. The team aimed to develop an image classification model for categorizing document types, facing challenges in data processing, model selection, and performance optimization. Their experience highlights the complexities of machine learning competitions and the strategies employed to improve model accuracy."
date: 2024-08-13 12:00:10 -0400
toc: true
toc_sticky: true
categories:
  - AIBootcamp
tags:
  - AIBootcamp 
  - 패스트캠퍼스 
  - 패스트캠퍼스AI부트캠프 
  - 업스테이지패스트캠퍼스 
  - UpstageAILab 
  - 국비지원 
  - 패스트캠퍼스업스테이지에이아이랩 
  - 패스트캠퍼스업스테이지부트캠프
---

# Digital Treasure Quest Team's Document Type Classification Challenge Journey

This document details the journey of the Digital Treasure Quest team as they participated in a document type classification competition. The team aimed to develop an image classification model for categorizing document types, facing challenges in data processing, model selection, and performance optimization. Their experience highlights the complexities of machine learning competitions and the strategies employed to improve model accuracy.

by 박석 / Park, Surk

## Competition Overview

The Document Type Classification competition challenged participants to develop a model that could classify given document images into one of 17 classes. This task is crucial for efficiently processing and automating large volumes of documents across various industries including finance, healthcare, insurance, and logistics.

### 1. Dataset Provided
- 1,570 document images belonging to 17 classes for training, and 3,140 document images for model performance validation.

### 2. Algorithms Allowed
- Participants could utilize various image classification algorithms including deep learning, transfer learning, and convolutional neural networks (CNN).

### 3. Modeling Goal
- Develop a model that predicts one of 17 classes when given a document image as input.

### 4. Submission Format
- Results to be submitted in CSV file format.

## Competition Timeline

The project spanned from July 30, 2024, to August 11, 2024. Key dates included:

1. **Competition Start**:  
   July 30, 2024 (Tuesday) 10:00 - Dataset distribution and competition commencement

2. **Team Merge Deadline**:  
   July 31, 2024 (Wednesday) 10:00 - Deadline for team formations

3. **Development Period**:  
   July 30, 2024 to August 10, 2024 - Time for model development and testing

4. **Final Submission**:  
   August 11, 2024 (Sunday) 19:00 - Deadline for submitting the final model

## Team Strategy and Culture

### Team Pros
- Diverse team with varied experiences, higher average age, and high receptivity to AI assistants.

### Team Cons
- Low experience in team-based R&D using Git, Python-based R&D, machine learning/deep learning R&D, and low domain knowledge related to the competition topic.

### Strategic Approach
- Utilize AutoML tools like DataRobot to guide feature engineering and model selection. Each team member to conduct machine learning modeling according to their individual skill levels.

### Team Culture
- Focus on individual learning, mutual respect, maximizing productivity through AI assistants, and ensuring at least one submission per team member without sacrificing individual schedules or resources.

## Data Description and Processing

The dataset consisted of training data with 1,570 images across 17 classes, and test data with 3,140 images. The team conducted Exploratory Data Analysis (EDA) to understand the differences between training and test images.

### Training Images
- Well-organized, good quality, standardized orientation, minimal noise or distortion.

### Test Images
- More varied quality, rotations, flips, some damaged images, varied lighting conditions, and shadows.

### Data Processing
- Implemented data augmentation and cleansing. Addressed mislabeled data issues, though limited due to unclear classification criteria from the competition organizers.

## Model Selection and Development

The team utilized DataRobot for initial model selection and focused on three top-performing machine learning models:

1. **Regularized Logistic Regression L2**  
   - Implemented with train-time image augmentation and a pretrained MobileNetV3-Small-Pruned Multi-Level Global Average Pooling Image Featurizer.

2. **Keras Slim Residual Neural Network Classifier**  
   - Utilized similar augmentation and featurizer as the logistic regression model, with a 1-layer, 64-unit neural network architecture.

3. **Baseline Image Classifier**  
   - Employed train-time image augmentation, grayscale downscaled image featurizer, and regularized logistic regression.

## Model Evaluation and Optimization

The team faced challenges in reducing the gap between Submit F1 Score and Test F1 Score. They implemented various strategies to improve model generalization:

1. **Cross-Validation**  
   - Implemented k-fold cross-validation to ensure model stability and prevent overfitting.

2. **Data Augmentation**  
   - Experimented with increasing data augmentation up to 20 times, though with limited success in reducing score discrepancies.

3. **Regularization Techniques**  
   - Applied learning schedulers, dropout, weight decay, and batch normalization to combat overfitting.

4. **Ensemble Methods**  
   - Explored ensemble techniques, including voting methods across fold models, to improve overall performance.

## Final Results and Lessons Learned

The Digital Treasure Quest team achieved a final rank of 10 in the competition leaderboard. Their top submission, "Ensemble_top9_v1", scored an F1 score of 0.9087.

### Key Takeaways
- The importance of data cleansing, the potential of ensemble techniques, and the need to focus on generalization performance rather than just public leaderboard scores.

### Future Strategies
- Consider more thorough data cleansing, explore advanced ensemble methods, and prioritize models with high generalization capabilities.

### Team Growth
- Despite initial limitations in R&D experience, the team demonstrated significant learning and adaptation throughout the competition.

---

# DataRobot Review

## Question 1: Advantages and Disadvantages of DataRobot

### Advantages

1. **Automated Modeling**:  
   DataRobot automates the entire process from data preparation to model development, evaluation, and deployment. This allows even non-experts to easily develop machine learning models.

2. **Support for Various Algorithms**:  
   DataRobot automatically tests multiple algorithms and selects the optimal model. It provides strong performance even for users with limited understanding of different algorithms.

3. **Model Interpretability**:  
   DataRobot offers transparency in explaining model results and how the model arrived at its conclusions, allowing users to trust and leverage the model in business decision-making.

4. **Scalability and Deployment**:  
   DataRobot is cloud-based and easily scalable, enabling model deployment across various environments, making it suitable for large-scale data and real-time applications.

5. **Integration of Multiple Data Sources**:  
   DataRobot supports the integration of various data sources, facilitating data preprocessing and modeling. It offers flexibility by allowing the use of diverse data formats.

### Disadvantages

1. **High Cost**:  
   DataRobot offers powerful features, but it comes at a high cost. This can be a burden for small businesses or organizations with limited budgets.

2. **Increased Dependency**:  
   Over-reliance on the platform's automated features can lead to models being developed without a deep understanding of machine learning, potentially leading to technical debt over time.

3. **Limited Customization**:  
   DataRobot focuses on automation, which may limit the fine-tuning needed for specific requirements. Advanced users might find the customization options insufficient.

4. **Complexity in Use**:  
   The interface may feel non-intuitive or complex, especially for beginners. The abundance of automated processes may lead to users not fully understanding the entire process.

5. **Data Security and Privacy Concerns**:  
   Being a cloud-based platform, there might be concerns regarding the security and privacy of sensitive data. There could be limitations in complying with certain industry or regulatory requirements.

## Question 2: Limitations and Disadvantages of DataRobot in Analyzing Unstructured Data

1. **Complexity in Processing Unstructured Data**:  
   Although DataRobot provides functionalities for processing unstructured data, the process can be more complex compared to structured data. When dealing with highly sophisticated unstructured data processing and feature engineering, the built-in capabilities of DataRobot might not be sufficient.

2. **Limited Customization**:  
   Handling unstructured data, particularly in text processing (NLP) or image analysis, may require custom functions. However, DataRobot’s emphasis on automation might restrict detailed adjustments for specific unstructured data processing needs, such as unique text preprocessing or image filtering techniques.

3. **Difficulty in Model Interpretation**:  
   In unstructured data analysis, interpreting model results becomes more challenging. While DataRobot emphasizes model interpretability, understanding the features or intermediate results generated from unstructured data can still be limited, especially when using complex deep learning models.

4. **Limitations in Data Preprocessing**:  
   Unstructured data often requires a significant preprocessing phase. DataRobot’s automated preprocessing features might be limited in specific scenarios, making it difficult to perform the detailed adjustments necessary for complex unstructured data preprocessing, such as specialized tokenization for text or specific preprocessing stages for image data.

5. **Restrictions in Advanced Unstructured Data Modeling**:  
   For advanced modeling of unstructured data (e.g., deep learning-based image classification, natural language processing), DataRobot’s capabilities might not be fully sufficient. When a project requires more complex model architectures or the latest research findings, the provided models or approaches in DataRobot may be limited.

## Question 3: Advantages and Disadvantages of Using NoCode AutoML Tools vs. Python-Based Tools for Team Projects

### NoCode AutoML Tools (e.g., DataRobot)

#### Advantages

1. **Rapid Prototyping and Productivity**:  
   NoCode tools automate processes such as data preparation, modeling, evaluation, and deployment, enabling quick results. Teams can develop prototypes rapidly and test ideas efficiently.

2. **Inclusion of Non-Experts**:  
   NoCode tools allow team members without machine learning expertise to easily contribute to model development, fostering active participation across diverse team backgrounds.

3. **Automated Model Selection and Optimization**:  
   These tools automatically test multiple algorithms and select the best model, saving teams time on optimization and delivering better performance more quickly.

#### Disadvantages

1. **Limited Customization**:  
   NoCode tools focus on automation, which can make it challenging to fine-tune models for specific requirements. Advanced modeling or unique needs might not be easily accommodated.

2. **Increased Technical Dependency**:  
   Over-reliance on these tools can result in a shallow understanding of machine learning principles and processes, potentially limiting the team’s technical capabilities in the long term.

3. **Cost Issues**:  
   Commercial NoCode tools often come with high costs. As project scope or complexity increases, the cost can become a significant burden, especially for teams with budget constraints.

### Python-Based Tools

#### Advantages

1. **High Flexibility and Customization**:  
   Python-based tools allow for fine-grained control at the code level, enabling teams to tailor the modeling process to specific project needs and requirements.

2. **Utilization of Open Source Ecosystem**:  
   Python offers a vast ecosystem of open-source libraries and community support. This provides access to a wide range of machine learning algorithms, data processing tools, and visualization libraries at no cost, while also enabling the adoption of the latest technologies.

3. **Learning and Skill Development**:  
   Working with Python-based tools helps team members develop a deeper understanding of machine learning principles and coding skills, enhancing the team’s technical capabilities in the long run.

#### Disadvantages

1. **Longer Development Time**:  
   Python-based projects often require more time due to the need for coding, debugging, and optimization. This can be particularly challenging when developing complex models.

2. **Steep Learning Curve**:  
   For team members unfamiliar with Python and machine learning libraries, the learning curve can be steep, potentially slowing down the initial stages of the project.

3. **Complexity in Collaboration Management**:  
   Code-based projects can involve complex collaboration management, including version control, code review, and integration testing, which can be time-consuming, especially for large teams.
