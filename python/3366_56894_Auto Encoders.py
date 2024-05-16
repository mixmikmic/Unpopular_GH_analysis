from yann.network import network
def convolutional_autoencoder ( dataset= None, verbose = 1 ):
    """
    This function is a demo example of a deep convolutional autoencoder. 
    This is an example code. You should study this code rather than merely run it.  
    This is also an example for using the deconvolutional layer or the transposed fractional stride
    convolutional layers.

    Args: 
        dataset: Supply a dataset.    
        verbose: Similar to the rest of the dataset.
    """
    dataset_params  = {
                            "dataset"   : dataset,
                            "type"      : 'x',
                            "id"        : 'data'
                    }

    visualizer_params = {
                    "root"       : '.',
                    "frequency"  : 1,
                    "sample_size": 32,
                    "rgb_filters": False,
                    "debug_functions" : False,
                    "debug_layers": True,  
                    "id"         : 'main'
                        }  
                      
    # intitialize the network    
    optimizer_params =  {        
                "momentum_type"       : 'nesterov',             
                "momentum_params"     : (0.65, 0.95, 30),      
                "regularization"      : (0.0001, 0.0001),       
                "optimizer_type"      : 'rmsprop',                
                "id"                  : "main"
                    }
    net = network(   borrow = True,
                     verbose = verbose )                       

    net.add_module ( type = 'datastream', 
                     params = dataset_params,
                     verbose = verbose )
    
    net.add_module ( type = 'visualizer',
                     params = visualizer_params,
                     verbose = verbose 
                    ) 
    net.add_module ( type = 'optimizer',
                     params = optimizer_params,
                     verbose = verbose )
    # add an input layer 
    net.add_layer ( type = "input",
                    id = "input",
                    verbose = verbose, 
                    origin = 'data', # if you didnt add a dataset module, now is 
                                                 # the time. 
                    mean_subtract = True )

    
    net.add_layer ( type = "conv_pool",
                    origin = "input",
                    id = "conv",
                    num_neurons = 20,
                    filter_size = (5,5),
                    pool_size = (1,1),
                    activation = 'tanh',
                    regularize = True,   
                    #stride = (2,2),                          
                    verbose = verbose
                    )

    net.add_layer ( type = "flatten",
                    origin = "conv",
                    id = "flatten",
                    verbose = verbose
                    )

    net.add_layer ( type = "dot_product",
                    origin = "flatten",
                    id = "hidden-encoder",
                    num_neurons = 1200,
                    activation = 'tanh',
                    dropout_rate = 0.5,                    
                    regularize = True,
                    verbose = verbose
                    )

    net.add_layer ( type = "dot_product",
                    origin = "hidden-encoder",
                    id = "encoder",
                    num_neurons = 128,
                    activation = 'tanh',
                    dropout_rate = 0.5,                        
                    regularize = True,
                    verbose = verbose
                    )

    net.add_layer ( type = "dot_product",
                    origin = "encoder",
                    id = "decoder",
                    num_neurons = 1200,
                    activation = 'tanh',
                    input_params = [net.dropout_layers['encoder'].w.T, None],
                    # Use the same weights but transposed for decoder. 
                    learnable = False,                    
                    # because we don't want to learn the weights of somehting already used in 
                    # an optimizer, when reusing the weights, always use learnable as False   
                    dropout_rate = 0.5,                                         
                    verbose = verbose
                    )           

    net.add_layer ( type = "dot_product",
                    origin = "decoder",
                    id = "hidden-decoder",
                    num_neurons = net.layers['flatten'].output_shape[1],
                    activation = 'tanh',
                    input_params = [net.dropout_layers['hidden-encoder'].w.T, None],
                    # Use the same weights but transposed for decoder. 
                    learnable = False,                    
                    # because we don't want to learn the weights of somehting already used in 
                    # an optimizer, when reusing the weights, always use learnable as False    
                    dropout_rate = 0.5,                                        
                    verbose = verbose
                    )                                            

    net.add_layer ( type = "unflatten",
                    origin = "hidden-decoder",
                    id = "unflatten",
                    shape = (net.layers['conv'].output_shape[2],
                             net.layers['conv'].output_shape[3],
                             20),
                    verbose = verbose
                    )

    net.add_layer ( type = "deconv",
                    origin = "unflatten",
                    id = "deconv",
                    num_neurons = 20,
                    filter_size = (5,5),
                    pool_size = (1,1),
                    output_shape = (28,28,1),
                    activation = 'tanh',
                    input_params = [net.dropout_layers['conv'].w, None],        
                    learnable = False,              
                    #stride = (2,2),
                    verbose = verbose
                    )

    # We still need to learn the newly created biases in the decoder layer, so add them to the 
    # Learnable parameters list before cooking

    net.active_params.append(net.dropout_layers['hidden-decoder'].b)
    net.active_params.append(net.dropout_layers['decoder'].b)    
    net.active_params.append(net.dropout_layers['deconv'].b)
    

    net.add_layer ( type = "merge",
                    origin = ("input","deconv"),
                    id = "merge",
                    layer_type = "error",
                    error = "rmse",
                    verbose = verbose)

    net.add_layer ( type = "objective",
                    id = "obj",
                    origin = "merge", # this is useless anyway.
                    layer_type = 'value',
                    objective = net.layers['merge'].output,
                    datastream_origin = 'data', 
                    verbose = verbose
                    )          

    learning_rates = (0.04, 0.0001, 0.00001)  
    net.cook( objective_layers = ['obj'],
              datastream = 'data',
              learning_rates = learning_rates,
              verbose = verbose
              )

    # from yann.utils.graph import draw_network
    # draw_network(net.graph, filename = 'autoencoder.png')    
    net.pretty_print()
    net.train( epochs = (10, 10), 
               validate_after_epochs = 1,
               training_accuracy = True,
               show_progress = True,
               early_terminate = True,
               verbose = verbose)

if __name__ == '__main__':
    import sys
    print " creating a new dataset to run through"
    from yann.special.datasets import cook_mnist_normalized_zero_mean as cook_mnist  
    data = cook_mnist (verbose = 2)
    dataset = data.dataset_location()

    convolutional_autoencoder ( dataset , verbose = 2 )

