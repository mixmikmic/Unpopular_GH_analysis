from yann.network import network
from yann.special.datasets import cook_mnist

data = cook_mnist()
dataset_params  = { "dataset": data.dataset_location(), "id": 'mnist', "n_classes" : 10 }

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

optimizer_params =  {
            "momentum_type"       : 'polyak',
            "momentum_params"     : (0.9, 0.95, 30),
            "regularization"      : (0.0001, 0.0002),
            "optimizer_type"      : 'rmsprop',
            "id"                  : 'polyak-rms'
                    }
net.add_module ( type = 'optimizer', params = optimizer_params )

learning_rates = (0.05, 0.01, 0.001)

net.cook( optimizer = 'polyak-rms',
          objective_layer = 'nll',
          datastream = 'mnist',
          classifier = 'softmax',
          )

net.train( epochs = (20, 20),
           validate_after_epochs = 2,
           training_accuracy = True,
           learning_rates = learning_rates,
           show_progress = True,
           early_terminate = True)

net.test()

from yann.pantry.tutorials.mlp import mlp
mlp(dataset = data.dataset_location())





