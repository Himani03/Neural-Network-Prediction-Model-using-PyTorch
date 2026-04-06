# Neural Network Prediction Model (PyTorch)

## Overview

This project focuses on predicting weekly retail sales using deep learning. I built a hybrid **CNN + LSTM model in PyTorch** to capture both feature relationships and time-based trends from historical retail data. The goal was to simulate a real-world demand forecasting system used in retail companies for inventory planning and business decision-making.

The project includes **data preprocessing, feature engineering, model training, evaluation, and visualization using Matplotlib**.


## Business Problem

Retail businesses often struggle with:

* Stock shortages
* Overstock inventory
* Seasonal demand spikes
* Promotion-based demand fluctuations

Accurate demand forecasting helps:

* Improve inventory planning
* Reduce operational cost
* Increase sales performance

This project predicts **Weekly Sales** using historical store data and external factors.


## Dataset

Dataset used: **Walmart Sales Forecasting Dataset**

Files used:

* train.csv
* test.csv
* features.csv
* stores.csv

The dataset contains:

* Store information
* Weekly sales
* Holiday indicators
* Economic indicators (CPI, unemployment)
* Promotional markdowns
* Store type and size

These features help the model learn real retail demand behavior.


## Project Structure

```
retail-ai-pytorch/
│
├── data/
├── notebooks/
│   └── retail_model.ipynb
│
├── src/
│   ├── model.py
│   ├── train.py
│   └── evaluation.py
│
├── plots/
│   ├── training_loss.png
│   └── prediction_vs_actual.png
│
├── requirements.txt
└── README.md
```


## Approach

### 1. Data Merging

The dataset was split across multiple files, so I merged:

* Sales data
* Store information
* External features

This allowed the model to learn from all relevant factors affecting retail sales.


### 2. Feature Engineering

Date column was converted into:

* Year
* Month
* Week

This helps the model capture **seasonal trends and weekly demand patterns**, which are very important in retail forecasting.


### 3. Handling Missing Values

Some promotional columns had missing values. These were replaced with zero, assuming no promotion occurred. This helped maintain dataset consistency and avoid training errors.


### 4. Encoding Categorical Variables

Store type was converted into numerical format using one-hot encoding. This allows the neural network to interpret categorical information properly.


### 5. Feature Scaling

StandardScaler was used to normalize features. Neural networks train faster and more reliably when input values are scaled.


## Model Architecture

The model combines **CNN and LSTM layers**:

### CNN Layer

The convolution layer helps detect relationships between features such as:

* Promotions
* Store performance
* Economic indicators

This improves feature extraction.


### LSTM Layer

Retail sales depend heavily on time patterns. LSTM helps capture:

* Seasonal demand
* Holiday spikes
* Weekly trends

This improves forecasting accuracy.


### Fully Connected Layer

The final layers generate **weekly sales prediction**.


## Training Process

* Data converted to PyTorch tensors
* Batch training using DataLoader
* Loss function: Mean Squared Error
* Optimizer: Adam
* Validation monitoring during training

Batch training was used to reduce memory usage and improve training stability.


## Training Visualization

Training and validation loss were plotted using Matplotlib to monitor model performance.

This helps identify:

* Model convergence
* Overfitting
* Training stability

Saved plot:

```
plots/training_loss.png
```


## Prediction Visualization

Model predictions were compared with actual values to evaluate performance.

This helps understand:

* Prediction quality
* Trend matching
* Forecast reliability

Saved plot:

```
plots/prediction_vs_actual.png
```


## Technologies Used

Programming:

* Python

Machine Learning:

* PyTorch
* CNN
* LSTM

Data Processing:

* Pandas
* NumPy
* Scikit-Learn

Visualization:

* Matplotlib
* Seaborn


## How to Run

Install dependencies:

```
pip install -r requirements.txt
```

Open notebook:

```
notebooks/retail_model.ipynb
```

Run cells sequentially.


## Results

* Stable training convergence
* Decreasing validation loss
* Good prediction trend alignment
* Generalized forecasting behavior


## Why This Project

This project demonstrates:

* End-to-end machine learning pipeline
* Retail demand forecasting
* PyTorch deep learning implementation
* Model evaluation and visualization
* Production-style project structure


## Future Improvements

* Hyperparameter tuning
* Add attention mechanism
* Try transformer-based forecasting
* Deploy model as API


