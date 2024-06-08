import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

def convert_keras_to_saved_model(keras_model_path, saved_model_path):
    # Load the Keras model
    model = tf.keras.models.load_model(keras_model_path)
    
    # Ensure the model is built by calling it with some example input
    if not model.built:
        print("model not built")
        example_input = tf.ones((1,) + model.input_shape[1:])  # Create an example input tensor
        model(example_input)

    # NOTE the output is invalid(for c++ prog at least)
    # for input_tensor in model.inputs:
    #     print("Input Tensor Name: ", input_tensor.name) # need to be add in the c++
    # for output_tensor in model.outputs:
    #     print("Output Tensor Name: ", output_tensor.name) # need to be add in the c++

    # Convert the Keras model to a concrete function
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
    
    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    
    # Save the frozen graph
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir=saved_model_path,
                      name="saved_model.pb",
                      as_text=False)
    
    print("Model has been converted and saved at:", saved_model_path)
    
# Define paths
keras_model_path = 'saved_model/my_model.keras'
saved_model_path = 'saved_model_pb'

# Convert the model
convert_keras_to_saved_model(keras_model_path, saved_model_path)



