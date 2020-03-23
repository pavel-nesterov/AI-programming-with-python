import argparse

def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type = str, default = 'flower_data/', help = 'path to the folder of flower images, just folder name, no slashes') 
    parser.add_argument('--save_dir', type = str, default = 'checkpoints/', help = 'Directory to save checkpoints') 
    parser.add_argument('--arch', type = str, default = 'vgg13', help = "Selected architecture") 
    parser.add_argument('--learning_rate', type = float, default = 0.01)
    parser.add_argument('--hidden_units', type = str, default = 512)
    parser.add_argument('--epochs', type = int, default = 30)
    parser.add_argument('--gpu', type = bool, default = False)
    
    in_args = parser.parse_args()
    #print(type(in_args.gpu))
    print("Argument 1:", in_args.data_dir)
    print("Argument 2:", in_args.arch)
    print("Argument 3:", in_args.learning_rate)
    return in_args

def get_hardcoded_input_args():
    data_dir = 'flower_data/' 
    save_dir = 'checkpoints/'
    arch = 'vgg13'
    learning_rate = 0.01
    hidden_units = 512
    epochs = 30
    gpu = False
    in_args = [data_dir, save_dir, arch, learning_rate, hidden_units, epochs, gpu]
    #print(in_args)
    return in_args


#print("Train aruments successfully imported")

#get_input_args()
#get_hardcoded_input_args()