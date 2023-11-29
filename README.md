# Investigation-of-Degradation-Patterns-from-KRR-to-Transformers
## Enhancing the Degradation Patterns Using Advanced

## Machine Learning Techniques

**Presented by: Aziz Ben Ali**

```
Referents: Alessio Gagliardi
Ioannis Kouroudis
```
```
Technical University of Munich
TUM School of Computation Information and Technology
```
**Munich, October 10th, 2023**


###### Measuring the output power of solar panels in real-world conditions can be

###### costly and time-consuming.

###### Understanding the mechanisms behind the degradation patterns of solar panels

###### is a highly complex task

## Motivation


## Approach


#### 4

## Data Cleaning and Processing

### 1) Outdoor Measurements

**Actual Outdoor Measurements**


## Data Cleaning and Processing

### 1) Indoor Measurements

**Unsmoothed Indoor Measurements**

**Smoothed Indoor Measurements**


## Training and Testing

**Cleaned Outdoor Measurements**


##### Multi-Layer Bidirectional LSTM-based Complex Neural Network

###### Fully connected Layers (FC)

###### Highway Layers (HL)

###### Bidirectional LSTM Layers (BILSTML)

###### This Model involves three main components:


**1) Exploring Fully connected Layers**

###### Each input node is connected to each output node.

###### During the training, FC experiences two main phases:

###### Forward pass

###### Backward pass

**Fully Connected Layers**

##### Multi-Layer Bidirectional LSTM-based Complex Neural Network


**A type of neural network.**

**The output is controlled by a gating mechanism.**

**allow the network to weather pass the input**

**as it is or apply transformations.**

**This process involves three main components: Transformation Gate**

**Nonlinear Transformation Function**

**Carry Gate**

**Highway Layer -1-**

- 1- https://paperswithcode.com/method/highway-layer

##### Multi-Layer Bidirectional LSTM-based Complex Neural Network

**2) Exploring Highway Layers**

#### 9


**A type of recurrent neural networks (RNNs).**

**Capable of learning long-term dependencies.**

**Involves 3 main components:**

###### a) LSTM Layers

**Forget Gate**

**Input Gate**

**Output Gate**

**LSTM layer -2-**

- 2- https://colah.github.io/posts/2015-08-Understanding-LSTMs/

**3) Long Short-Term Memory (LSTM) and Bidirectional LSTM Networks**

##### Multi-Layer Bidirectional LSTM-based Complex Neural Network

#### 10


Same machanism as LSTM.

Able of capturing relevant information and

relationships within the data in forward and

backward directions.

BILSTM layer -3-

- 3- https://www.baeldung.com/cs/bidirectional-vs-unidirectional-lstm

###### b) BILSTM Layers

3) Long Short-Term Memory (LSTM) and Bidirectional LSTM Networks

##### Multi-Layer Bidirectional LSTM-based Complex Neural Network

#### 11


4 BILSTML organized in a stack

capture of complex sequential pattern

Initialization of BILSTM Layers: The use of Kaiming–Normal Initialization

effective training convergence and less sensitivity to hyperparameters

For each BILSTM Layer, a corresponding Highway layer is applied.

enhances information flow throughout the network.

mitigates the problem of Vanishing Gradient.

Problem: Exploding Gradient => Gradient Clipping

4) Model Architecture

##### Multi-Layer Bidirectional LSTM-based Complex Neural Network


###### 3 FCs reduces the dimensionality while extracting

###### the relevant features.

###### We employ Dropout technique mitigates overfitting

###### problem

Dropout technique

4) Model Architecture

##### Multi-Layer Bidirectional LSTM-based Complex Neural Network


5) Results

##### Multi-Layer Bidirectional LSTM-based Complex Neural Network


###### The architecture of TST involves two main components:

###### Encoder: Positional Encodings

Multi-Head Self-Attention:

Position-wise Feedforward Network

Residual Connections and Layer Normalization

###### Decoder: Positional Encodings

Masked Multi-Head Self-Attention

Position-wise Feedforward Network

Residual Connections and Layer Normalization

Output Layer

Time Series Transformer -4-

- 4- https://machinelearningmastery.com/the-transformer-model/

##### Time Series Transformer (TST)

#### 15


##### Time Series Transformer (TST)

**5) Results**


###### Belongs to the category of ensemble

###### learning.

###### Employs gradient boosting.

###### Uses decision trees as its weak learners.

###### Each new tree is trained to minimize the

###### error made by the previous trees.

**XGBoost**

https://www.researchgate.net/figure/Flow-chart-of-XGBoost_fig3_

##### XGBoost

#### 17


```
Degradation Accelerators Mean Squared Error (MSE)
1.0 Sun in NI 1.
1.4 Sun in NI 1.
1.0 Sun in Air 1.
1.0 Sun in NI + 1.4 Sun in NI 1.
1.0 Sun in Air + 1.4 Sun in NI 1.
1.0 Sun in Air + 1.0 Sun in NI 1.
1.0 Sun in NI +1.4 Sun in NI +1.
Sun in Air 1.
```
```
Total Error Average 1.
```
##### XGBoost

**5) Results**


###### The prior over function:

###### Likelihood:

###### By applying Bayes’ theorem, GPR combines the prior and likelihood to calculate the

###### posterior distribution over functions p(f (x)|y)

###### GPR is a non-parametric, probabilistic, and bayesian approach to regression,

###### characterized by two main concepts:

##### Gaussian Process Regressor (GPR)


```
Degradation Accelerators Mean Squared Error (MSE)
1.0 Sun in NI 1.
1.4 Sun in NI 2.
1.0 Sun in Air 1.
1.0 Sun in NI + 1.4 Sun in NI 2.
1.0 Sun in Air + 1.4 Sun in NI 1.
1.0 Sun in Air + 1.0 Sun in NI 1.
1.0 Sun in NI +1.4 Sun in NI +1.
Sun in Air 1.
```
```
Total Error Average 1.
```
##### Gaussian Process Regressor (GPR)

**5) Results**


###### Decomposes the discrete time series into its constituent frequencies in the

###### frequency domain.

###### => We add the magnitude in decibels of the transformed output as an

###### independent feature.

###### Discrete Fourier Transformation (DFT)

##### Gaussian Process Regressor (GPR)

###### Optimization


**Degradation Accelerators Mean Squared Error (MSE)**
1.0 Sun in NI 1.06
1.4 Sun in NI 1.00

1.0 Sun in Air 1.06

1.0 Sun in NI + 1.4 Sun in NI 0.94
1.0 Sun in Air + 1.4 Sun in NI 1.06
1.0 Sun in Air + 1.0 Sun in NI 1.05
1.0 Sun in NI +1.4 Sun in NI +1.0
Sun in Air 1.04

Total Error Average 1.03

###### Results

##### Gaussian Process Regressor (GPR)

###### Discrete Fourier Transformation (DFT)


**KRR is a powerful machine learning model designed for regression tasks.**

**Unlike traditional methods that map data to high-dimensional feature spaces (computationally**

**expensive), KRR uses the "kernel trick”.**

**allows KRR to work with inner products of mapped data points directly using a kernel**

**function**

**The central idea is to find optimal coefficients that minimize a dual objective function:**

**by applying the kernel function and combining the results with the learned α coefficient, KRR can**

**predict the target values of new inputs.**

##### Kernel Ridge Regression (KRR)


```
Degradation Accelerators Mean Squared Error (MSE)
1.0 Sun in NI 1.35
1.4 Sun in NI 1.11
1.0 Sun in Air 1.30
1.0 Sun in NI + 1.4 Sun in NI 1.34
1.0 Sun in Air + 1.4 Sun in NI 1.26
1.0 Sun in Air + 1.0 Sun in NI 1.39
1.0 Sun in NI +1.4 Sun in NI +1.0
Sun in Air 1.34
```
```
Total Error Average 1.29
```
##### Kernel Ridge Regression (KRR)

###### Results


**Widely recognized hyperparameter tuning**

**technique**

**Works as following:**

###### Grid Search

**Define the Search Space (Grid)**

**Use Cross-Validation**

**Select the Best Model**

**Train and evaluate the best Model**

**More powerful hyperparameter technique than**

**Grid Search.**

**Involves three key concepts:**

**Objective Function**

**Surrogate Function (Gaussian**

**Process)**

**Acquisition Function**

###### Bayesian Optimization

###### Total Error Average: 1.01 Total Error Average: 0.74

##### Kernel Ridge Regression (KRR)

###### Optimization


## Discussion


# Conclusion


###### Wavelet

```
Approximation Coefficients (Aj):
These coefficients represent the low-frequency components of the
signal at scale j.
Detail Coefficients (Dj):
These coefficients represent the high-frequency components of the
signal at scale j.
```
```
powerful method for analyzing time-series data across different scales or frequencies
=> No need to the use of padding.
```
###### Decomposition into Components:

```
Extract and use the detail coefficients as features for indoor measurements.
Total error average: 1,11 > 1,03
```
## Discussion
