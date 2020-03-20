import argparse

def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path_to_file', type = str, help = 'path to the folder of image to predict') 
    parser.add_argument('checkpoint', type = str, help = 'Checkpoint name with model to use for prediction') 
    parser.add_argument('--top_k', type = int, default = 3, help = "Number of top most probable classes") 
    parser.add_argument('--category_names', type = str, help = "Name of the file with mapping between numerical category and category name")
    parser.add_argument('--gpu', type = bool, default = False)
    
    in_args = parser.parse_args()
    print(in_args)
    print ("Class probabilities: ______")

    return in_args

get_input_args()