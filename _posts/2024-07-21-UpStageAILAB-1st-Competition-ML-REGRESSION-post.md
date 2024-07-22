---
title: "UpStage AI LAB 1st Competition - ML REGRESSION"
excerpt: "We introduce the journey of the ‘Digital Treasure Quest’ team as they took on the ‘Apartment Actual Price Prediction’ contest. Our team consists of 5 members with diverse backgrounds and experiences, and we took on the task of developing a model to predict the actual transaction price of apartments in Seoul using AI technology. Let’s take a look at our challenge process, results, and lessons learned along the way."
date: 2024-07-21 12:00:10 -0400
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

# Digital Treasure Quest Team's Journey in the Apartment Price Prediction Competition

## 1. Competition Info

### 1.1. Overview

#### Competition Overview

##### Goal
Develop a model to predict apartment prices in Seoul based on real transaction data.

##### Introduction
The House Price Prediction competition aims to develop a model that predicts real transaction prices of apartments in Seoul using provided data. Apartment prices are influenced by various factors such as proximity to rivers, parks, and shopping centers. The predictive model considers these factors to accurately forecast market prices and assist in real estate transactions.

##### Provided Datasets
1. **Apartment transaction data**: Includes location, size, year of construction, nearby facilities, and transportation convenience.
2. **Subway station information**: Provided by Seoul City.
3. **Bus stop information**: Provided by Seoul City.
4. **Evaluation data**: For validating model performance.

##### Available Algorithms
Participants can use various regression algorithms such as linear regression, decision trees, random forests, and deep learning.

##### Modeling Goal
Develop an accurate and generalized model to forecast market trends and support real estate decisions. Participants gain practical experience by evaluating model performance and understanding the correlations between various features.

##### Submission Format
Submit results as a CSV file with 9,272 rows of predicted apartment transaction prices.

### 1.2. Timeline

#### Project Duration
- **July 9 (Tue) 10:00 ~ July 19 (Fri) 19:00**

#### Key Milestones
- **Competition Start**: July 9 (Tue) 10:00
- **Team Merge Deadline**: July 10 (Wed) 10:00
- **Development and Testing Period**: July 9 (Tue) 10:00 ~ July 18 (Thu) 19:00
- **Final Model Submission**: July 19 (Fri) 19:00

#### Detailed Schedule
1. **July 9 (Tue)**: Dataset release and competition start
   - Begin data exploration and preprocessing
   - Form teams and allocate roles
2. **July 10 (Wed)**: Team merge deadline
   - Start initial model development
   - Complete data preprocessing
3. **July 11 (Thu)**: Model validation
   - Train models and analyze initial results
   - Discuss ways to improve model performance
4. **July 12 (Fri)**: Model improvement
   - Try various algorithms
   - Tune parameters
5. **July 13 (Sat)**: Model testing
   - Evaluate model performance using validation data
   - Analyze and correct errors
6. **July 14 (Sun)**: Feedback implementation
   - Modify and improve model based on feedback
   - Integrate additional data
7. **July 15 (Mon)**: Optimization
   - Optimize the model and maximize performance
   - Perform additional feature engineering
8. **July 16 (Tue)**: Review results
   - Review and test the final model
   - Begin analyzing and documenting results
9. **July 17 (Wed)**: Documentation
   - Document model development process and results
   - Final review and corrections
10. **July 18 (Thu)**: Final checks
    - Final model inspection and submission preparation
    - Final testing and results verification
11. **July 19 (Fri)**: Final model submission and competition end
    - Submit the final model
    - Prepare for results announcement

### 1.3. Evaluation

#### Evaluation Method

The competition is a regression task to predict the actual transaction prices of apartments. The models developed by participants are evaluated using **RMSE (Root Mean Squared Error)**.

##### What is RMSE?
RMSE measures the average deviation between predicted and actual values. It is calculated by taking the square root of the average squared differences between predicted and actual values.

##### Evaluation Criteria
- **Prediction Accuracy**: How well the model captures the difference between actual and predicted prices.
- **Low RMSE Value**: A lower RMSE indicates better predictive performance.

##### Calculation Method
![RMSE](./images/rmse-desc.png)

##### Context
In the context of apartment transactions, RMSE quantitatively reflects how closely the regression model's predictions match actual transaction prices, playing a crucial role in assessing the model's performance.

## 2. Winning Strategy

### 2.1. DTQ Team's Pros and Cons
#### Pros
- Team members with diverse experience
- High average age
- High acceptance of AI assistants

#### Cons 
- Low experience in team-based R&D using Git
- Low experience in Python-based R&D
- Low experience in machine learning/deep learning R&D
- Limited domain knowledge related to the competition topic

### 2.2. DTQ Team's Strategic Approach
- Utilize AutoML tools like DataRobot to guide feature engineering and model selection.
- Each team member develops different machine learning models based on their expertise.

### 2.3. DTQ Team's Culture & Spirit
- The goal of participating in the competition is to gain knowledge and experience in machine learning R&D through individual learning.
- Understand and respect each team member's situation and contributions.
- Actively use AI assistants to maximize individual productivity.
- Avoid sacrificing individual schedules or resources for the team's overall goal.
- Ensure that each team member makes at least one submission.

## 3. Components

### 3.1. Directory

- code: Source code and related documents for each team member's experiments
- docs: Team documents (presentations, reference materials)
- images: Attached images
- README.md: Digital Treasure Quest team's Apartment Price Prediction competition journey

## 4. Data Description

### 4.1. Dataset Overview

#### Training Data
- File name: train.csv
- Rows: 1,118,822
- Features: 52
- Numeric: 22
- Text: 5
- Categorical: 18
- Date: 7
- Size: 244 MB

#### Prediction Data
- File name: test.csv
- Rows: 9,272
- Features: 51
- Numeric: 22
- Text: 5
- Categorical: 17
- Date: 7
- Size: 2.46 MB

### 4.2. EDA

#### Feature Description

*Detailed descriptions of features, excess zeros, outliers, disguised missing values, inliers, and target leakage.*

### 4.3. Feature Engineering

#### Methods Considered
- One-Hot Encoding
- Missing Values Imputed
- Smooth Ridit Transform
- Binning of numerical variables
- Matrix of char-grams occurrences using tfidf

#### Feature Selection
*Selected input features and target feature details.*

## 5. Modeling

### 5.1. Model Selection

#### Model Validation Stability
To prevent overfitting, models are validated using k-fold cross-validation and a holdout sample.

#### Data Partition Methodology
- Main: Random sampling for data partition.
- Additional: Time-series-based sampling based on UpStage AI Lab mentoring insights.

#### Selected Models
- eXtreme Gradient Boosted Trees Regressor
- Keras Slim Residual Network Regressor
- Light Gradient Boosted Trees Regressor

### 5.2. eXtreme Gradient Boosted Trees Regressor (DataRobot)

#### Modeling Descriptions
*Detailed description of the Gradient Boosting Machines and XGBoost.*

#### Modeling Process

##### Hyperparameters
*Details of hyperparameters and their best values.*

##### Feature Impact
*Feature impact visualization.*

##### Word Cloud
*Word cloud visualization.*

### 5.3. Keras Slim Residual Network Regressor (DataRobot)

#### Modeling Descriptions
*Description of neural networks and the specific Keras Slim Residual Network model.*

#### Modeling Process

##### Neural Network
*Neural network structure visualization.*

##### Hyperparameters
*Details of hyperparameters and their best values.*

##### Training
*Training process visualization.*

##### Feature Impact
*Feature impact visualization.*

##### Word Cloud
*Word cloud visualization.*

### 5.4. Light Gradient Boosted Trees Regressor (DataRobot)

#### Modeling Descriptions
*Description of LightGBM and its benefits.*

#### Modeling Process

##### Hyperparameters
*Details of hyperparameters and their best values.*

##### Feature Impact
*Feature impact visualization.*

##### Word Cloud
*Word cloud visualization.*

### 5.5. PyTorch Residual Network Regressor (By 박석)

#### Modeling Descriptions
*Description of using PyTorch for Residual Network modeling.*

#### Modeling Process

##### Hyperparameters
*Details of hyperparameters and their best values.*

##### Training
*Training process visualization.*

##### Trial History
*Details of each trial with changes and results.*

##### Additional Trial Shared with UpStage AI Stages Community
*Links to shared insights and trials.*

### 5.6. XGBoost (By 백경탁)

#### Daily Log
*Daily logs documenting key activities and insights.*

##### 2024.07.16
*Selection of important features.*

##### 2024-07-17
*Imputation of missing values and data handling.*

##### 2024-07-19
*Log transformation of target values and time-series splitting.*

#### Trial History
*Details of each trial with changes and results.*

#### Mentoring List
*Key mentoring insights and actions taken.*

### 5.7. Light GBM (By 한아름)

#### Trial History
*Baseline code comparisons and results of Light GBM model with Time Series K-Fold data splitting.*

#### Mentoring List
*Key mentoring insights and actions taken.*

### 5.8. Baseline Code Enhancement (By 이승현)
*TBD*

## 6. Result

### Leader Board

#### Final - Rank 3
*Leaderboard final position and insights.*

#### Submit History
*Submission history with insights and learning points.*

### Presentation
*Presentation details.*

## Personal Retrospective

### **Personal Retrospective Writing Guidelines**

- The personal retrospective in the report aims to summarize the technical challenges attempted during the competition, lessons learned during the learning process, limitations and challenges faced, etc., to reflect on what and how I tried as a learner and what I gained.
- In practice, it's important not only to do the work well but also to communicate logically and continue to grow based on continuous records. We hope you become effective workers through AI technology learning!
- Example:
  - **How did I achieve my learning goals?**
    - What were my team's and my learning goals?
    - Personal learning aspects
    - Collaborative learning aspects
  - **How did I improve the model?**
    - Knowledge and techniques used
  - **What did I achieve and realize as a result of my actions?**
  - **What new changes did I attempt compared to before, and what were the effects?**
  - **What were the limitations and regrets?**
  - **What will I try newly in the next competition based on the limitations/lessons learned?**

### Retrospective

During the competition, my primary goal was to apply and enhance my skills in data analysis and machine learning, with a focus on predicting apartment prices accurately. Our team aimed to leverage diverse experiences and the strengths of various algorithms to achieve this.

#### What I Did to Achieve My Learning Goals
- **Team Learning Goals**: Our team's main goal was to build a robust predictive model using advanced machine learning techniques.
- **Personal Learning Goals**: My personal objective was to deepen my understanding of feature engineering and model optimization.

#### How I Improved the Model
- Utilized One-Hot Encoding and Smooth Ridit Transform for categorical variables.
- Applied rigorous feature selection to enhance model performance.
- Experimented with various algorithms, including eXtreme Gradient Boosted Trees and Light Gradient Boosted Trees.

#### Achievements and Realizations
- Successfully reduced RMSE through iterative model tuning.
- Gained insights into the impact of different features on model predictions.
- Improved my ability to preprocess and handle large datasets efficiently.

#### New Changes and Their Effects
- Implemented Time Series K-Fold data splitting which improved the model's generalization capability.
- Introduced additional feature engineering techniques that enhanced prediction accuracy.

#### Limitations and Regrets
- Faced challenges in balancing model complexity and interpretability.
- Limited time to explore all potential feature interactions.

#### Plans for Future Competitions
- In the next competition, I plan to focus more on automated feature selection techniques.
- I will also explore more advanced neural network architectures to capture complex patterns in the data.

Overall, this competition was a valuable learning experience, providing practical insights into machine learning model development and the importance of teamwork in data science projects.
