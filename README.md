# Doodle Classifier  
*A web-based deep learning application for recognizing hand-drawn sketches using TensorFlow, developed for the course of Artificial Intelligence @ FCSE.*

## Deep Learning Model  
The model is built using **TensorFlow** and **Keras** and is trained on the [Google Quick, Draw! Dataset](https://quickdraw.withgoogle.com/data). The architecture consists of:  

- **Convolutional Layers (Conv2D)**: Extract spatial features from sketches  
- **Batch Normalization**: Stabilizes training and speeds up convergence  
- **MaxPooling Layers**: Reduces feature map size and prevents overfitting  
- **Dense Layers**: Fully connected layers for final classification  
- **Dropout Layer**: Regularization to avoid overfitting  

### Model Training  
- **Dataset**: 20 categories from the Quick, Draw! dataset  
- **Image Size**: 28x28 grayscale images  
- **Optimizer**: Adam (`learning_rate=0.0005`)  
- **Loss Function**: Sparse Categorical Crossentropy  
- **Metrics**: Accuracy  
- **Epochs**: 8  
- **Batch Size**: 64  

## Web Application  
The application is built using:  
- **Flask** (Backend API for model inference)  
- **JavaScript & HTML/CSS** (Frontend with a drawing canvas)  
- **Fetch API** (Sends the drawn sketch as base64 image for prediction)
