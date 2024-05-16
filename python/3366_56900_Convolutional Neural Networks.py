from yann.network import network
from yann.utils.graph import draw_network
from yann.special.datasets import cook_mnist
def lenet5 ( dataset= None, verbose = 1, regularization = None ):             
    """
    This function is a demo example of lenet5 from the infamous paper by Yann LeCun. 
    This is an example code.  
    
    Warning:
        This is not the exact implementation but a modern re-incarnation.

    Args: 
        dataset: Supply a dataset.    
        verbose: Similar to the rest of the dataset.
    """
    optimizer_params =  {        
                "momentum_type"       : 'nesterov',             
                "momentum_params"     : (0.65, 0.97, 30),      
                "optimizer_type"      : 'rmsprop',                
                "id"                  : "main"
                        }

    dataset_params  = {
                            "dataset"   : dataset,
                            "svm"       : False, 
                            "n_classes" : 10,
                            "id"        : 'data'
                      }

    visualizer_params = {
                    "root"       : 'lenet5',
                    "frequency"  : 1,
                    "sample_size": 144,
                    "rgb_filters": True,
                    "debug_functions" : False,
                    "debug_layers": False,  # Since we are on steroids this time, print everything.
                    "id"         : 'main'
                        }       

    # intitialize the network
    net = network(   borrow = True,
                     verbose = verbose )                       
    
    # or you can add modules after you create the net.
    net.add_module ( type = 'optimizer',
                     params = optimizer_params, 
                     verbose = verbose )

    net.add_module ( type = 'datastream', 
                     params = dataset_params,
                     verbose = verbose )

    net.add_module ( type = 'visualizer',
                     params = visualizer_params,
                     verbose = verbose 
                    )
    # add an input layer 
    net.add_layer ( type = "input",
                    id = "input",
                    verbose = verbose, 
                    datastream_origin = 'data', # if you didnt add a dataset module, now is 
                                                 # the time. 
                    mean_subtract = False )
    
    # add first convolutional layer
    net.add_layer ( type = "conv_pool",
                    origin = "input",
                    id = "conv_pool_1",
                    num_neurons = 20,
                    filter_size = (5,5),
                    pool_size = (3,3),
                    activation = ('maxout', 'maxout', 2),
                    # regularize = True,
                    verbose = verbose
                    )

    net.add_layer ( type = "conv_pool",
                    origin = "conv_pool_1",
                    id = "conv_pool_2",
                    num_neurons = 50,
                    filter_size = (3,3),
                    pool_size = (1,1),
                    activation = 'relu',
                    # regularize = True,
                    verbose = verbose
                    )      


    net.add_layer ( type = "dot_product",
                    origin = "conv_pool_2",
                    id = "dot_product_1",
                    num_neurons = 1250,
                    activation = 'relu',
                    # regularize = True,
                    verbose = verbose
                    )

    net.add_layer ( type = "dot_product",
                    origin = "dot_product_1",
                    id = "dot_product_2",
                    num_neurons = 1250,                    
                    activation = 'relu',  
                    # regularize = True,    
                    verbose = verbose
                    ) 
    
    net.add_layer ( type = "classifier",
                    id = "softmax",
                    origin = "dot_product_2",
                    num_classes = 10,
                    # regularize = True,
                    activation = 'softmax',
                    verbose = verbose
                    )

    net.add_layer ( type = "objective",
                    id = "obj",
                    origin = "softmax",
                    objective = "nll",
                    datastream_origin = 'data', 
                    regularization = regularization,                
                    verbose = verbose
                    )
                    
    learning_rates = (0.05, .0001, 0.001)  
    net.pretty_print()  
    draw_network(net.graph, filename = 'lenet.png')   

    net.cook()

    net.train( epochs = (20, 20), 
               validate_after_epochs = 1,
               training_accuracy = True,
               learning_rates = learning_rates,               
               show_progress = True,
               early_terminate = True,
               patience = 2,
               verbose = verbose)

    print(net.test(verbose = verbose))
data = cook_mnist()
dataset = data.dataset_location()
lenet5 ( dataset, verbose = 2)

