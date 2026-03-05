from .utils.data_loader import *
from .ann.neural_network import *
from .ann.optimizers import *
import wandb

"""
Inference Script
Evaluate trained models on test sets
"""

import argparse

def parse_arguments():
    """
    Parse command-line arguments for inference.
    
    TODO: Implement argparse with:
    - model_path: Path to saved model weights(do not give absolute path, rather provide relative path)
    - dataset: Dataset to evaluate on
    - batch_size: Batch size for inference
    - hidden_layers: List of hidden layer sizes
    - num_neurons: Number of neurons in hidden layers
    - activation: Activation function ('relu', 'sigmoid', 'tanh')
    """
    parser = argparse.ArgumentParser(description='Run inference on test set')

    parser.add_argument("-d","--dataset",choices=["mnist","fashion_mnist"])
    parser.add_argument("-e","--epochs")
    parser.add_argument("-b","--batch")
    parser.add_argument("-l","--loss",choices=["cross_entropy","mse"])
    parser.add_argument("-o","--optimizer",choices=["sgd","momentum","nag","RMSprop"])
    parser.add_argument("-lr","--learning_rate")
    parser.add_argument("-wd","--weight_decay")
    parser.add_argument("-nhl","--num_layers")
    parser.add_argument("-sz","--hidden_size")
    parser.add_argument("-a","--activation",choices=["relu","sigmoid","tanh"])
    parser.add_argument("-w_i","--weight_init")
    parser.add_argument("-wp","--wandb_project")
    parser.add_argument("-msp","--model_path")
    
    return parser.parse_args()


def load_model(model_path):
    data = np.load(model_path, allow_pickle=True)
    return data.item()


def evaluate_model(model, X_test, y_test): 
    """
    Evaluate model on test data.
    """
    return model.evaluate(X_test,y_test)


def main():
    """
    Main inference function.
    """

    print("initialised")
    args = parse_arguments()
    print(vars(args))


    X_train,Y_train,x_test,y_test = load_data(args.dataset)
    # print(Y_train)
    ann = NeuralNetwork(args,len(X_train[0]),10)

    wt = load_model(args.model_path)

    ann.set_weights(wt)

    training = ann.evaluate(X_train,Y_train)
    test = ann.evaluate(x_test,y_test)

    metrics = {f"train/{k}": v for k, v in training.items() if k != "logits"}
    metrics.update({f"test/{k}": v for k, v in test.items() if k != "logits"})

    wandb.log(metrics)
    
    print(metrics)
    
    print("Evaluation complete!")


if __name__ == '__main__':
    main()
