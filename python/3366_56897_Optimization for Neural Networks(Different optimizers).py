from yann.network import network
from yann.special.datasets import cook_mnist
import matplotlib.pyplot as plt

def get_cost():
    costs = []
    with open('./resultor/costs.txt') as costf:
        costs = [float(cost.rstrip()) for cost in costf]
    return costs

def plot_costs(costs, labels):
    for cost, label in zip(costs, labels):
        plt.plot(cost,label=label)
    plt.legend()
    plt.show()
    
costs = []

data = cook_mnist()
dataset_params  = { "dataset": data.dataset_location(), "id": 'mnist', "n_classes" : 10 }
def mlp(dataset_params, optimizer_params, optimizer_id):
    net = network()
    net.add_layer(type = "input", id ="input", dataset_init_args = dataset_params)
    net.add_layer (type = "dot_product",
                   origin ="input",
                   id = "dot_product_1",
                   num_neurons = 800,
                   regularize = True,
                   activation ='relu')

    net.add_layer (type = "dot_product",
                   origin ="dot_product_1",
                   id = "dot_product_2",
                   num_neurons = 800,
                   regularize = True,
                   activation ='relu')
    net.add_layer ( type = "classifier",
                    id = "softmax",
                    origin = "dot_product_2",
                    num_classes = 10,
                    activation = 'softmax',
                    )

    net.add_layer ( type = "objective",
                    id = "nll",
                    origin = "softmax",
                    )
    
    net.add_module ( type = 'optimizer', params = optimizer_params )
    learning_rates = (0.05, 0.01, 0.001)
    net.cook( verbose = 0,
             optimizer = optimizer_id,
              objective_layer = 'nll',
              datastream = 'mnist',
              classifier = 'softmax',
              )
    net.train(verbose=0,
              epochs = (20, 20),
           validate_after_epochs = 2,
           training_accuracy = True,
           learning_rates = learning_rates,
           show_progress = True,
           early_terminate = True)
    return net

optimizer_params =  {
                "momentum_type"       : 'false',
                "momentum_params"     : (0.9, 0.95, 30),
                "regularization"      : (0.0001, 0.0002),
                "optimizer_type"      : 'sgd',
                "id"                  : 'sgd'
                        }
net = mlp(dataset_params, optimizer_params, 'sgd')
costs.append(get_cost())

labels = ['no momentum']
plot_costs(costs, labels)

optimizer_params =  {
                "momentum_type"       : 'polyak',
                "momentum_params"     : (0.9, 0.95, 30),
                "regularization"      : (0.0001, 0.0002),
                "optimizer_type"      : 'sgd',
                "id"                  : 'sgd-polyak'
                        }
net = mlp(dataset_params, optimizer_params, 'sgd-polyak')
costs.append(get_cost())

labels = ['no momentum', 'polyak']
plot_costs(costs, labels)

optimizer_params =  {
                "momentum_type"       : 'nesterov',
                "momentum_params"     : (0.9, 0.95, 30),
                "regularization"      : (0.0001, 0.0002),
                "optimizer_type"      : 'sgd',
                "id"                  : 'sgd-nesterov'
                        }
net = mlp(dataset_params, optimizer_params, 'sgd-nesterov')
costs.append(get_cost())

labels = ['no momentum', 'polyak', 'nesterov']
plot_costs(costs, labels)

optimizer_params =  {
                "momentum_type"       : 'false',
                "regularization"      : (0.0001, 0.0002),
                "optimizer_type"      : 'adagrad',
                "id"                  : 'adagrad'
                        }
net = mlp(dataset_params, optimizer_params, 'adagrad')
costs.append(get_cost())

labels = ['sgd','sgd-polyak','sgd-nestrov', 'adagrad']
plot_costs(costs, labels)

optimizer_params =  {
                "momentum_type"       : 'false',
                "regularization"      : (0.0001, 0.0002),
                "optimizer_type"      : 'rmsprop',
                "id"                  : 'rmsprop'
                        }
net = mlp(dataset_params, optimizer_params, 'rmsprop')
costs.append(get_cost())

labels = ['sgd','sgd-polyak','sgd-nestrov', 'adagrad','rmsprop']
plot_costs(costs, labels)

optimizer_params =  {
                "momentum_type"       : 'false',
                "regularization"      : (0.0001, 0.0002),
                "optimizer_type"      : 'adam',
                "id"                  : 'adam'
                        }
net = mlp(dataset_params, optimizer_params, 'adam')
costs.append(get_cost())

labels = ['sgd','sgd-polyak','sgd-nestrov', 'adagrad','rmsprop','adam']
plot_costs(costs, labels)

