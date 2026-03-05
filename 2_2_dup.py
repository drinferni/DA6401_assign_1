import wandb
import subprocess
import sys

# 1. Define the sweep configuration
sweep_config = {
    'method': 'random', 
    'metric': {
        'name': 'val_acc',  # FIXED: Matches your wandb.log key
        'goal': 'maximize'   
    },
    'parameters': {
        'dataset': {'values': ['mnist']},
        'epochs': {'values': [5, 10,20,40]},
        'batch': {'values': [32, 64, 128]},
        'loss': {'values': ['cross_entropy', 'mse']},
        'optimizer': {'values': ['sgd', 'momentum', 'nag', 'adam','RMSprop' ,'nadam']},
        'learning_rate': {'distribution': 'log_uniform_values', 'min': 1e-4, 'max': 1e-1},
        'num_layers': {'values': [1, 2, 3]},
        'hidden_size': {'values': [16, 32, 64]},
        'activation': {'values': ['relu', 'sigmoid', 'tanh']},
        'weight_init': {'values': ['random', 'xavier']}
    }
}


def train_logic():
    # IMPORTANT: We do NOT call wandb.init() here because src.train does it.
    # We use wandb.init() temporarily just to GET the config from the agent,
    # then we finish it immediately so the subprocess can start its own.
    run = wandb.init()
    c = run.config
    
    cmd = [
        sys.executable, "-m", "src.train",
        "-d", str(c.dataset),
        "-e", str(c.epochs),
        "-b", str(c.batch),
        "-l", str(c.loss),
        "-o", str(c.optimizer),
        "-lr", str(c.learning_rate),
        "-nhl", str(c.num_layers),
        "-sz", str(c.hidden_size),
        "-a", str(c.activation),
        "-w_i", str(c.weight_init),
        "-wp", "DA_assign_1",
        "-msp", f"model_{run.id}.txt"
    ]
    
    # Close this dummy run so the subprocess doesn't collide with it
    run.finish()
    
    # Run the actual training script
    subprocess.run(cmd)

if __name__ == "__main__":
    # To run 100 runs sequentially, simply set count=100 and do not use multiprocessing.
    # This will run one after another in a single process.
    wandb.agent("gxqmm8rn", function=train_logic, project="DA_assign_1", count=21)