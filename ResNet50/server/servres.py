import numpy as np
import flwr as fl
from flwr.common import Parameters, Scalar
from flwr.server import client_proxy
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Flatten, Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from keras.models import Sequential
from keras.applications.resnet import ResNet50
from keras.callbacks import ReduceLROnPlateau
from glob import glob
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"]="-1" 


base = ResNet50(weights = 'imagenet', include_top = False, input_shape = (180, 180, 3))
tf.keras.backend.clear_session()

for layer in base.layers:
    layer.trainable = False
    
model = Sequential()
model.add(base)
model.add(GlobalAveragePooling2D())
model.add(BatchNormalization())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

optm = Adam(lr=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=optm, 
                  metrics=['accuracy'])



class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[fl.common.Parameters, Dict[str, Any]]:
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(rnd, results, failures)

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        weights, metrics = super().aggregate_fit(rnd, results, failures)

        weights_list = fl.common.parameters_to_ndarrays(weights)

        if weights is not None:

            print(f"Saving round {rnd} weights...")

            # Load the model weights
            model.set_weights(weights_list)

            #model.save(f"model_round_{rnd}.h5")
            model.save(f"model_round_{rnd}.h5")
            
        return weights, metrics

def fit_config(rnd: int) -> Dict[str, str]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epoch_global": str(rnd),
        "epochs": str(1),
        "batch_size": str(32),
    }
    return config

def get_eval_fn(model):
    """Return an evaluation function for server-side evaluation."""

    val_gen = ImageDataGenerator(rescale = 1./255)
    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    test_set = val_gen.flow_from_directory('Data\\val',
                                      target_size = (180,
                                                     180),
                                      batch_size = 1,
                                      class_mode = 'categorical')
    x_test, y_test = next(test_set)

    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(parameters)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(test_set)
        return loss, {"accuracy": accuracy}

    return evaluate

def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.
    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5 if server_round < 4 else 10
    return {"val_steps": val_steps}

strategy = SaveModelStrategy(
    fraction_fit=1.0,
    min_fit_clients=2,
    min_available_clients=2,
    evaluate_fn=get_eval_fn(model),
    on_fit_config_fn=fit_config,
    on_evaluate_config_fn=evaluate_config,
    )



# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:50051",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy,
)