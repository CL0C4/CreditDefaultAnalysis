# Credit Risk and Loss Optimization Model

An end-to-end risk analytics pipeline using the [Default of Credit Card Clients Dataset](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset?resource=download).  
It tests how varying credit approval thresholds directly impact a logistic regression model's ability to adequately identify a lendee's risk of defaulting. It can further be used to evaluate how an institution's risk mitigation, customer acquisition friction, and bottom-line portfolio profitability can change by changing this threshold.



## Features

- Utilizes a [Logistic Regression Model](https://www.geeksforgeeks.org/machine-learning/understanding-logistic-regression/) to calculate the probability of a default based on Credit Utilization and Payment Ratio
- Implemented and documented the results using 3 different thresholds to ascertain how a stricter, more risk-averse threshold affects customer approval rates, as well as False Positives(Unnecessary Rejections) and False Negatives(Missed Defaults)
- Uses StandardScaler to ensure that naturally larger numbers are not necessarily treated has more important (i.e. ensures that credit limit, which could scale to 100,000+, is not immediately deemed more important than age, which will realistically only approach ~80)


## Results and Interpretations  
### 0.50 Threshold  
- 870 False Negatives: The model approved 422 borrowers who ended up defaulting  
- 1878 False Positives: The model flagged 4117 good borrowers   
- 56% Recall: The model caught 56% of the actual defaulters, more accurate than not but 56% leaves the lender very exposed to credit losses  
- 37% Precision: Out of everyone flagged by the model, only 37% actually defaulted, meaning the threshold for the model is highly risk-averse  
- 0.45 F1-Score: Indicates the model is struggling to find a healthy mix between keeping default numbers down and approving safe accounts  
- 0.69 ROC-AUC Score: The model has a moderate ability to rank risks, i.e the model will assign higher-risk probability to the correct individual 69% of the time  
- 69% Accuracy: Looks good, however as previously mentioned, the amount of non-defaulters far outranks defaulters, so a model could just guess "no default" for everyone and be quite accurate, so this will be ignored  

### 0.30 Threshold  
- 422 False Negatives: The model approved 422 borrowers who ended up defaulting  
- 4117 False Positives: The model flagged 4117 good borrowers  
- 79% Recall: The model caught 79% of the actual defaulters, providing much stronger portfolio protection, though 21% still leaves some exposure to credit losses  
- 28% Precision: Out of everyone flagged by the model, only 28% actually defaulted, meaning this lower threshold is extremely risk-averse and heavily penalizes applicant approval rates  
- 0.41 F1-Score: Indicates the model is still struggling to find a healthy mix, dropping slightly because the aggressive gain in catching defaults was offset by a massive wave of false alarms  
- 0.69 ROC-AUC Score: The model has a moderate ability to rank risks, i.e the model will assign higher-risk probability to the correct individual 69% of the time (unchanged as sorting power remains constant across thresholds)  
- 50% Accuracy: Looks significantly worse on paper, however as previously mentioned, raw accuracy is deceptive, and this drop is to be expected as the lower threshold seeks to mitigate risk, not maximize overall correctness  

### 0.15 Threshold
- 160 False Negatives: The model approved 160 borrowers who ended up defaulting  
- 5888 False Positives: The model flagged 5888 good borrowers   
- 92% Recall: The model caught 92% of the actual defaulters, providing maximum portfolio protection against catastrophic credit losses, leaving only minimal exposure  
- 24% Precision: Out of everyone flagged by the model, only 24% actually defaulted, meaning this extremely aggressive threshold heavily restricts lending volume and over-penalizes safe applicants  
- 0.38 F1-Score: Indicates the model is struggling to find a healthy mix, dropping further because the massive wave of false alarms outweighed the excellent gains in catching defaults  
- 0.69 ROC-AUC Score: The model has a moderate ability to rank risks, i.e the model will assign higher-risk probability to the correct individual 69% of the time (unchanged as sorting power remains constant across thresholds)  
- 33% Accuracy: As mentioned above, this drop is to be expected as it rejects far more applicants than higher thresholds  

## Software and Libraries used

### Python
#### Pandas
Pandas is a data analysis and manipulation tool and was used in this project to alter the formatting of the dataset in order to be used for the Logistic Regression Model
```py
>>> pip install pandas
```
#### Numpy
Numpy is used for mathematical operations and scientific computing. It was used in conjuction with Pandas to handle the small amount of NaN and infinity values added to the Pandas dataframe when cleaning the data
```py
>>> pip install numpy
```
#### Scikit-learn
Scikit-learn is a machine learning library for Python. It provided the backbone of this project, with the following functions being used:
- StandardScaler
- Train Test Split
- Classification Report
- ROC_AUC Score
- Confusion Matrix
```py
>>> pip install scikit-learn
```
#### Imbalanced Learn
Imbalanced Learn relies on Scikit-learn to handle unbalanced datasets. It was used in this project for SMOTE (Synthetic Minority Oversampling Technique)
```py
>>> pip install imblearn
```

### Authors

My name is [Christopher LoCascio](https://github.com/CL04A). I am an undergraduate student at the [University of Arkansas at Fayetteville](https://www.uark.edu/) pursuing a B.S in Computer Science and a B.A in Economics. This project serves as an introduction into data analytics and machine learning



