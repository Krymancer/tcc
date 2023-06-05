import tensorflow as tf

def main():
    checkpoint = tf.train.load_checkpoint('checkpoints/model.42-0.16.h5')
    variable_to_shape_map = checkpoint.get_variable_to_shape_map()

    for variable_name in variable_to_shape_map:
        print(variable_name, variable_to_shape_map[variable_name])

        
if __name__ == '__main__':
    main()