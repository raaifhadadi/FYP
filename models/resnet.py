import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import metrics

def resnet_block(X, filters, kernels=[5, 3], index=0):
        
        X_shortcut = X
        
        X = keras.layers.Conv1D(filters=filters[0], kernel_size=kernels[0], padding='same', name=f'conv_{index}_1')(X)
        X = keras.layers.BatchNormalization(name=f'bn_{index}_1')(X)
        X = keras.layers.Activation('relu')(X)
        
        X = keras.layers.Conv1D(filters=filters[1], kernel_size=kernels[1], padding='same', name=f'conv_{index}_2')(X)
        X = keras.layers.BatchNormalization(name=f'bn_{index}_2')(X)
        
        X = keras.layers.concatenate([X, X_shortcut])
        X = keras.layers.Activation('relu')(X)
        
        return X
    
def model(X, num_classes=5, filters = [16, 16], kernels = [5, 3], layers=4, hidden_units=128):
        
        X = keras.layers.Conv1D(filters=filters[0], kernel_size=kernels[0], input_shape=X.shape[1:], name='conv1')(X)
        X = keras.layers.BatchNormalization(name='bn1')(X)
        X = keras.layers.Activation('leaky_relu')(X)
        
        X = keras.layers.MaxPooling1D(2)(X)
        
        for i in range(layers):
            X = resnet_block(X, filters, kernels=kernels, index=i+1)
            
            if i == 1:
                X = keras.layers.MaxPooling1D(2)(X)
        
        features_output = keras.layers.GlobalAveragePooling1D()(X)
        X = keras.layers.Dropout(0.5)(features_output)
        
        X = keras.layers.Dense(hidden_units, activation='leaky_relu', name='dense1')(X)
        X = keras.layers.Dropout(0.1)(X)
        X = keras.layers.Dense(hidden_units/2, activation='leaky_relu', name='dense2')(X)
        X = keras.layers.Dense(num_classes, activation='sigmoid', name='output')(X)
        
        return X, features_output