from utils.data_loader import *
from ann.neural_network import *
from ann.optimizers import *
import wandb

"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse

def parse_arguments():
    """
    Parse command-line arguments.
    
    TODO: Implement argparse with the following arguments:
    - dataset: 'mnist' or 'fashion_mnist'
    - epochs: Number of training epochs
    - batch_size: Mini-batch size
    - learning_rate: Learning rate for optimizer
    - optimizer: 'sgd', 'momentum', 'nag', 'rmsprop'
    - hidden_layers: List of hidden layer sizes
    - num_neurons: Number of neurons in hidden layers
    - activation: Activation function ('relu', 'sigmoid', 'tanh')
    - loss: Loss function ('cross_entropy', 'mse')
    - weight_init: Weight initialization method
    - wandb_project: W&B project name
    - model_save_path: Path to save trained model (do not give absolute path, rather provide relative path)
    """
    parser = argparse.ArgumentParser(description='Train a neural network')

#     -d, --dataset: Choose between mnist and fashion mnist.
# 2. -e, --epochs: Number of training epochs.
# 3. -b, --batch size: Mini-batch size.
# 4. -l, --loss: Choice of mean squared error or cross entropy.
# 5. -o, --optimizer: One of sgd, momentum, nag, rmsprop, adam, nadam.
# 6. -lr, --learning rate: Initial learning rate.
# 7. -wd, --weight decay: Weight decay for L2 regularization.
# 8. -nhl, --num layers: Number of hidden layers.
# 9. -sz, --hidden size: Number of neurons in each hidden layer (list of values).
# 10. -a, --activation: Choice of sigmoid, tanh, relu for every hidden layer.
# 11. -w i, --weight init: Choice of random or xavier.

    parser.add_argument("-d","--dataset",choices=["mnist","fashion_mnist"])
    parser.add_argument("-e","--epochs")
    parser.add_argument("-b","--batch_size")
    parser.add_argument("-l","--loss",choices=["cross_entropy","mse"])
    parser.add_argument("-o","--optimizer",choices=["sgd","momentum","nag","adam","nadam","RMSprop"])
    parser.add_argument("-lr","--learning_rate")
    parser.add_argument("-wd","--weight_decay")
    parser.add_argument("-nhl","--num_layers",dest="num_hidden_layers")
    parser.add_argument("-sz", "--hidden_size", dest="hidden_layer_sizes", nargs="+",  type=int)
    parser.add_argument("-a","--activation",choices=["relu","sigmoid","tanh"])
    parser.add_argument("-w_i","--weight_init")
    parser.add_argument("-wp","--wandb_project")
    parser.add_argument("-msp","--model_save_path")
    
    return parser.parse_args()


def main():
    """
    Main training function.
    """
    print("initialised")
    args = parse_arguments()
    print(vars(args))


    X_train,Y_train,x_test,y_test = load_data(args.dataset)
    args.n = len(X_train[0])
    args.m = 10
    ann = NeuralNetwork(args)
    for x in ann.weight_mat:
        print(x.shape)

    print(len(X_train))
    ann.train(X_train,Y_train, int(args.epochs), int(args.batch_size))

    run = None

    if args.wandb_project :
        run = wandb.init(project=args.wandb_project, config=vars(args))
    
    training = ann.evaluate(X_train,Y_train)
    test = ann.evaluate(x_test,y_test)

    metrics = {f"train/{k}": v for k, v in training.items() if k != "logits"}
    metrics.update({f"test/{k}": v for k, v in test.items() if k != "logits"})

    if run != None:
        wandb.log(metrics)


    if run != None:
        run.finish()

    best_weights = ann.get_weights()
    np.save("best_model.npy", best_weights)

    # ann.print_mat(ann.gradient_mat)
    
    print("Training complete!")


if __name__ == '__main__':
    main()
