import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import metrics

def model(X, num_classes=5, filters = [12, 18, 24], kernels = [5,5,3], hidden_units=[128, 64], dropout=[0.5, 0.1]):
    
      
    X = keras.layers.Conv1D(filters=filters[0], kernel_size=kernels[0], input_shape=X.shape[1:])(X)
    
    for i in range(len(filters)-1):
        X = keras.layers.BatchNormalization()(X)
        X = keras.layers.Activation('leaky_relu')(X)
        X = keras.layers.MaxPooling1D(2)(X)
        X = keras.layers.Conv1D(filters=filters[i+1], kernel_size=kernels[i+1])(X)
        
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.Activation('leaky_relu')(X)
    
    features_output = keras.layers.GlobalAveragePooling1D()(X)
    X = features_output
    
    assert len(hidden_units) == len(dropout), 'Number of hidden_units layers and dropout layers must be equal'
    
    for i in range(len(hidden_units)):
        X = keras.layers.Dropout(dropout[i])(X)
        X = keras.layers.Dense(hidden_units[i], activation='leaky_relu')(X)
        
    X = keras.layers.Dense(num_classes, activation='sigmoid')(X)
    
    return X, features_output