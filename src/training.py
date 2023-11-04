from src.utils.common import read_config
from src.utils.data_mgmt import get_data
from src.utils.model import create_model,create_plots,save_model
import argparse 
import os 

def training(config_path):
    config = read_config(config_path)
    VALIDATION_DATA = config["params"]["validation_data"]
    (x_train,y_train),(x_valid,y_valid),(x_test,y_test)=get_data(VALIDATION_DATA)
    LOSS_FUNCTION= config["params"]["loss_function"]
    OPTIMIZER= config["params"]["optimizer"]
    METRICS = config["params"]["metrics"]
    NUM_CLASS = config["params"]["num_classes"]
    INPUT_SHAPE = config["params"]["input_shape"]

    model = create_model(LOSS_FUNCTION,OPTIMIZER,METRICS,NUM_CLASS,INPUT_SHAPE)

    EPOCHS = config["params"]["epochs"]
    VALIDATION = (x_valid,y_valid)
    history = model.fit(x_train,y_train,epochs=EPOCHS,validation_data=VALIDATION)
    create_plots(history)

    model_name = config["artifacts"]["model_name"]
    artifacts_dir = config["artifacts"]["artifacts_dir"]
    model_dir = config["artifacts"]["model_dir"]
    model_dir_path = os.path.join(artifacts_dir, model_dir)
    os.makedirs(model_dir_path,exist_ok=True)
    save_model(model,model_name,model_dir_path)


    save_model(model,model_name,model_dir_path)

    

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config","-c",default="config.yaml")

    parsed_args = args.parse_args()

    training(config_path=parsed_args.config)

