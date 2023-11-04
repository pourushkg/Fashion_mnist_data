import tensorflow as tf 

def get_data(validation_data):
    (x_train_full, y_train_full),(x_test, y_test)=tf.keras.datasets.fashion_mnist.load_data()
    x_valid,x_train=x_train_full[:validation_data]/255,x_train_full[validation_data:]/255
    y_valid,y_train=y_train_full[:validation_data],y_train_full[validation_data:]
    x_test=x_test/255

    return (x_train,y_train),(x_valid,y_valid),(x_test,y_test)