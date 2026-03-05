import numpy as np
from .neural_layer import *
from .objective_functions import *
from .optimizers import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import wandb
"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""

class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """

    num_layer:int
    hidden_size:int
    activation:any
    optimization:any
    weight_init:any
    lr:any
    dataset:any
    loss:any

    weight_mat:any
    gradient_mat:any
    z_val_mat:any
    a_mat:any
    loss_mat:any

    input_size:any
    output_size:any

    optimizer:any
    run:any

    
    def __init__(self, cli_args,n,m):
        """
        Initialize the neural network.

        Args:
            cli_args: Command-line arguments for configuring the network
        """
        print(cli_args)
        self.num_layer = int(cli_args.num_layers)
        self.hidden_size = int(cli_args.hidden_size)
        self.activation = cli_args.activation
        self.optimization = cli_args.optimizer
        self.weight_init = cli_args.weight_init
        self.lr = float(cli_args.learning_rate)
        self.datset = cli_args.dataset
        self.loss = cli_args.loss
        self.epoch = cli_args.epochs

        self.input_size = n
        self.output_size = m

        self.weight_mat = weight_initialsize(self.num_layer, self.hidden_size, self.weight_init,self.input_size,self.output_size)
        self.gradient_mat = []
        self.a_mat = []
        self.z_val_mat = []
        self.loss_mat = []
        self.optimizer = None

        self.grad_W = []
        self.grad_b = []

        self.run = wandb.init(project=cli_args.wandb_project, config=vars(cli_args))


        pass

    def print_mat(self,mat):
        for x in mat:
            print(x)
        print("\n")

    def debug(self):
        for x in self.weight_mat:
            print(x.shape)
            print(x)
        self.print_mat(self.z_val_mat)
        self.print_mat(self.a_mat)
        self.print_mat(self.loss_mat)
        self.print_mat(self.gradient_mat)
    
    def forward(self, X):
        """
        Forward propagation through all layers.
        
        Args:
            X: Input data
            
        Returns:
            Output logits
        """

        in_mat = X
        self.a_mat.append(np.array(in_mat))
        self.z_val_mat.append(np.array(in_mat))
        out_mat = in_mat
        for x in self.weight_mat:
            # print("meow")
            in_mat = np.insert(in_mat,0,1)
            z,out_mat = forward_layer(x,self.activation,in_mat)
            in_mat = out_mat
            self.a_mat.append(np.array(out_mat))
            self.z_val_mat.append(np.array(z))
        

        return out_mat
    
    def backward(self, y_true, y_pred):
        """
        Backward propagation to compute gradients.
        
        Args:
            y_true: True labels
            y_pred: Predicted outputs
            
        Returns:
            return grad_w, grad_b
        """

        l = len(self.weight_mat)
        # print(l,len(self.z_val_mat))
        l_mat = []
        loss_out = get_derivation_loss(y_true,y_pred,self.loss) * get_derivative_activation(self.activation,self.z_val_mat[l])
        l_mat.append(loss_out)
        for x in range(l-1,0,-1):
            # print(x)
            loss_vec = backward_layer(self.weight_mat[x],l_mat[(l-1)-x],self.z_val_mat[x],self.activation,x)
            l_mat.append(loss_vec)
        
        self.loss_mat = l_mat[::-1]

        # print(len(self.loss_mat),len(self.a_mat),len(self.z_val_mat))
        for x in range(0,len(self.loss_mat)):
            # print(self.loss_mat[x].shape,"uwu",self.a_mat[x].shape)
            a = self.a_mat[x]
            a = np.array(np.insert(a,0,1)).reshape(-1,1)
            b = self.loss_mat[x].reshape(-1,1)
            if x != len(self.loss_mat)-1:
                b = b[1:]
            # print(a.shape,"tT",b.shape)
            self.gradient_mat.append(a @ b.T)

        l = len(self.gradient_mat)

        self.grad_W = []
        self.grad_b = []

        for x in range(0,l,1):
            self.grad_W.append(self.gradient_mat[x][1:])
            self.grad_b.append(self.gradient_mat[x][:1])

        return self.grad_W[::-1],self.grad_b[::-1]

    
    def update_weights(self):
        """
        Update weights using the optimizer.
        """
        pass
    
    def train(self, X_train, y_train, epochs , batch_size):
        """
        Train the network for specified epochs.
        """

        if self.optimization.upper() == "SGD":
            self.optimizer = SGD(lr=self.lr)
        elif self.optimization.upper() == "MOMENTUM":
            self.optimizer = Momentum(lr=self.lr, beta=getattr(self, 'beta', 0.9))
        elif self.optimization.upper() == "NAG":
            self.optimizer = NAG(lr=self.lr, beta=getattr(self, 'beta', 0.9))
        elif self.optimization.upper() == "RMSPROP":
            self.optimizer = RMSprop(lr=self.lr, beta=getattr(self, 'beta', 0.99))


        for x in range (0,epochs):
            print("epochs =>" , x)
            l = len(X_train)
            count = 0
            cycle = (l + batch_size -1)//batch_size
            for y in range (0,cycle):
                new_X = X_train[count:count+batch_size]
                new_Y = y_train[count:count+batch_size]
                self.weight_mat = self.optimizer.preprocess(self.weight_mat)
                gradient_collector = None
                for dp in range(0,len(new_X)):
                    self.gradient_mat.clear()
                    self.a_mat.clear()
                    self.z_val_mat.clear()
                    self.loss_mat.clear()
                    pred_Y = self.forward(new_X[dp])
                    self.backward(new_Y[dp],pred_Y)
                    # for x in self.gradient_mat:
                    #     print(x.shape)
                    #     print(x)
                    if gradient_collector == None:
                        gradient_collector = [x.copy() for x in self.gradient_mat]
                    else :
                        for x in range(0, len(gradient_collector)):
                            gradient_collector[x] += self.gradient_mat[x]
                count += batch_size
                # print(len(gradient_collector))
                # for x in gradient_collector:
                #     print("gradient",x.shape)
                #     print(x)
                gradient_collector = [g / len(new_X) for g in gradient_collector]

                self.weight_mat = self.optimizer.update(self.weight_mat,gradient_collector)

    def test(self,X):
        y_pred = []

        for x in X:
            temp = self.forward(x)
            y_pred.append(temp)
        return y_pred
    

    def evaluate(self, X, y):
        """
        Evaluate the network on both training and validation/test data.
        """

        pred_probs = self.test(X)
        val_loss = get_loss(y, pred_probs, self.loss)
        
        pred_labels = np.argmax(pred_probs, axis=1)
        if len(y.shape) > 1 and y.shape[1] > 1:
            true_labels = np.argmax(y, axis=1)
        else:
            true_labels = y

        acc       = accuracy_score(true_labels, pred_labels)
        precision = precision_score(true_labels, pred_labels, average='macro', zero_division=0)
        recall    = recall_score(true_labels, pred_labels, average='macro', zero_division=0)
        f1        = f1_score(true_labels, pred_labels, average='macro', zero_division=0)
        
        return {
            "logits": pred_probs,
            "loss": val_loss,
            "acc": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1}


    def get_weights(self):
        d = {}
        for i, layer in enumerate(self.weight_mat):
            d[f"W{i}"] = self.weight_mat[i][1:].copy()
            d[f"b{i}"] = self.weight_mat[i][:1].copy()
        return d

    def set_weights(self, weight_dict):
        for i in range(0,(len(weight_dict)//2)):
            w_key = f"W{i}"
            b_key = f"b{i}"
            W = weight_dict[w_key]
            b = weight_dict[b_key]
            self.weight_mat[i] = np.concatenate((b, W), axis=0)
            

