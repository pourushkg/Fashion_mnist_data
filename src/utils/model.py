import tensorflow as tf 
import time 
import os 
import matplotlib.pyplot as plt 
import pandas as pd 


def create_model(LOSS_FUNCTION,OPTIMIZER,METRICS,NUM_CLASS,INPUT_SHAPE):
    layers=[
        tf.keras.layers.Flatten(input_shape=INPUT_SHAPE,name="input_layer"),
        tf.keras.layers.Dense(300,activation="relu",name="first_hidden_layer"),
        tf.keras.layers.Dense(100,activation="relu",name="second_hidden_layer"),
        tf.keras.layers.Dense(NUM_CLASS,activation="softmax",name="output_layer")
    ]

    model_clf= tf.keras.models.Sequential(layers)

    model_clf.summary()

    model_clf.compile(loss=LOSS_FUNCTION,
                      optimizer=OPTIMIZER,
                      metrics=METRICS)
    
    return model_clf

def create_plots(history):
    pd.DataFrame(history.history).plot(figsize=(10,7))
    plt.show()

def get_unique_file_name(filename):
    unique_file_name = time.strftime(f"%Y%m%d-%H%M%S{filename}")
    return unique_file_name

def save_model(model,model_name,model_dir):
    unique_file_name=get_unique_file_name(model_name)
    path_to_model = os.path.join(model_dir,unique_file_name)

    model.save(path_to_model)