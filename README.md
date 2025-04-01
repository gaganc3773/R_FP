# **Rainfall Prediction using Neural Networks**
## **Overview**
This project focuses on predicting rainfall using a **fully connected neural network (FCNN)** built with **PyTorch**. The dataset, sourced from a Kaggle competition, includes meteorological features like `temperature, humidity, wind speed, pressure, and historical rainfall records`. The model preprocesses data by handling missing values and normalizing features before training. It is optimized using the **Binary Cross-Entropy (BCE) loss function** and **AUC loss function**, ensuring better classification of rainy vs. non-rainy conditions. Model performance is evaluated using **AUC-ROC**.
## **Dataset**
- The dataset consists of train and test data in CSV file format, sourced from the Kaggle Playground Series.
- Features: `pressure`, `maxtemp`, `temperature`, `mintemp`, `dewpoint`, `humidity`, `cloud`, `sunshine`, `winddirection`, `windspeed`.
- Target: `rainfall` (binary target indicating whether rainfall occurred).
## **Model Architecture**
### **$1.$ Input Processing**
- Loading the data from CSV files.
- Conversion of the data into Tensors.
- Standardization of column values.
### **Neural Network Structure**
- `Input Layer`:Accepts processed meteorological data.  
- `Hidden Layers`: Fully connected layers with `ReLU activation`.  
- `Output Layer`: Single neuron with `Sigmoid activation` for binary classification.  \

The network is mathematically represented as:  
    $Y_{\theta}(x) = \sigma(W_n \cdots \text{ReLU}(W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2) + b_n)$
Where,
- $x$ → Weather feature input
- $\theta(W, b)$ → Weights and biases
- $\sigma$ → Sigmoid activation for binary classification
### **$3.$ Loss Computation**

The model is optimized using Binary Cross-Entropy Loss combined with AUC Loss:

$L = \text{BCE Loss} + \text{AUC Loss}$

where BCE Loss is computed as:

$L_{BCE} = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log \hat{y}^{(i)} + (1 - y^{(i)}) \log (1 - \hat{y}^{(i)}) \right]$

Where ACE Loss is computed as:

$L_{AUC} = 1 - \text{AUC}$

$\text{AUC} = \frac{1}{|P| \cdot |N|} \sum_{i \in P} \sum_{j \in N} \mathbb{I}(s_i > s_j)$
Where, 
- P: Set of data points labeled as positive.
- N: Set of data points labeled as negative.
- $\mathbb{I}(s_i > s_j)$: Indicator Function.

### **$4.$ Backward Propagation & Optimization**
- Use `Adam's` Optimizer to update weights $\theta$.
- The optimization step follows:  
  $\theta = \theta - \eta \frac{\partial \mathcal{L}}{\partial \theta}$  
  where **$\eta$** is the learning rate.

### **$5.$ Training Process**
- Train the model on the labeled dataset.
- Monitor training loss and AUC performance.
- performance using precision, recall, and accuracy

## **Features**
- **Preprocessing**: Data `normalization` and `feature standardization`.
- **Deep Learning Model**: `FCNN` with `multiple hidden layers.`
- **Optimization**: Uses `Binary Cross-Entropy Loss` + `AUC Loss` with `Adam optimizer`.
- **Evaluation**: Computes accuracy, precision, recall, and AUC score.

## References

- Walter Reade and Elizabeth Park. "Binary Prediction with a Rainfall Dataset." 2025. Available at: [https://kaggle.com/competitions/playground-series-s5e3](https://kaggle.com/competitions/playground-series-s5e3).
