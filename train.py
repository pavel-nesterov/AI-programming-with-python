
import argparse
import torch
from torchvision import datasets, transforms, models
#import train
from torch import nn
from torch import optim
import time
#from workspace_utils import active_session
#import helper
#import import_the_checkpoint
from collections import OrderedDict
import save_load_model
from workspace_utils import active_session 



def get_input_args():
    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument('--data_dir', type = str, default = 'aipnd-project_original udacity folder/flowers', help = 'path to the folder of flower images, just folder name, no slashes') 
        parser.add_argument('--save_dir', type = str, default = 'checkpoints/', help = 'Directory to save checkpoints with "/" at the end') 
        parser.add_argument('--arch', type = str, default = 'vgg11_bn', help = "Selected architecture") 
        parser.add_argument('--learning_rate', type = float, default = 0.001)
        parser.add_argument('--hidden_units', type = str, default = '3136, 784, 416', help = "3 comma-separated numbers for hidden layers input sizes, like \"3136, 784, 416\"")
        parser.add_argument('--epochs', type = int, default = 10)
        parser.add_argument('--gpu', type = bool, default = True)
    
        in_args = parser.parse_args()
        #print(type(in_args.gpu))
        #print("Argument 1:", in_args.data_dir)
        #print("Argument 2:", in_args.arch)
        #print("Argument 3:", in_args.learning_rate)
        in_args_dict = vars(in_args)
        in_args_list = [v for v in in_args_dict.values()]
        return in_args_list
   # stuff only to run when not called via 'import' here
    else:
        args = get_hardcoded_input_args()
        return(args)
    
    
    

def get_hardcoded_input_args():
    data_dir = 'aipnd-project_original udacity folder/flowers' 
    save_dir = 'checkpoints/'
    arch = 'vgg11_bn'
    learning_rate = 0.001
    hidden_units = '3136, 784, 416'
    epochs = 10
    gpu = True
    in_args = [data_dir, save_dir, arch, learning_rate, hidden_units, epochs, gpu]
    #print(in_args)
    return in_args

def define_transfroms(args):
    #print(args)
    data_dir = args[0]
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    
    
    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    validation_and_testing_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


    # TODO: Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_dataset = datasets.ImageFolder(valid_dir, transform=validation_and_testing_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=validation_and_testing_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(validation_dataset, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64)
    #print("Three loaders were defined in train.py")
    class_to_idx = train_dataset.class_to_idx
    #print("class_to_idx = ", class_to_idx)
    return trainloader, validloader, testloader, class_to_idx


def create_model(args):
    
    #models_arch_list = ['alexnet', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'squeezenet1_0', 'squeezenet1_1', 'densenet121', 'densenet169', 'densenet161', 'densenet201', 'googlenet', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', 'mobilenet_v2', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2', 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3', 'fcn_resnet50', 'fcn_resnet101', 'deeplabv3_resnet50', 'deeplabv3_resnet101']
    #models_arch_list = ['alexnet', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn']
    #arch = "vgg11"
    #model_arch_code = "model = models." + arch + "(pretrained=True)"
    #exec(model_arch_code)
    #print("model_arch_code = ", model_arch_code)
    
    
    #for model_arch in models_arch_list:
    #    model_arch_code = "model = models." + model_arch + "(pretrained=True)"
    #    print(model_arch)
    #    print(exec(model_arch_code))
        
        
    arch = args[2]

    
    if arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        #print(model.parameters)
        #classifier_name = 'classifier'
        classifier_input_size = 9216
    elif arch == 'vgg11':
        model = models.vgg11(pretrained=True)
        #print(model.parameters)
        classifier_input_size = 25088
    else:
        model = models.vgg11_bn(pretrained=True)
        #print(model.parameters)
        classifier_input_size = 25088
    
    #model = models.vgg13(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    hidden_units = args[4]
    hidden_units = hidden_units.split(', ') 
    hidden_units = [int(i) for i in hidden_units]
    #print(type(hidden_units[0]), hidden_units)

    classifier = nn.Sequential(OrderedDict([
                          ('dropout1', nn.Dropout(p=0.2, inplace=False)),
                          ('fc1', nn.Linear(classifier_input_size, hidden_units[0])),
                          ('relu1', nn.ReLU()),
                          ('dropout2', nn.Dropout(p=0.2, inplace=False)),
                          ('fc2', nn.Linear(hidden_units[0], hidden_units[1])),
                          ('relu2', nn.ReLU()),
                          ('dropout3', nn.Dropout(p=0.2, inplace=False)),
                          ('fc3', nn.Linear(hidden_units[1], hidden_units[2])),
                          ('relu3', nn.ReLU()),
                          ('dropout4', nn.Dropout(p=0.2, inplace=False)),
                          ('fc4', nn.Linear(hidden_units[2], 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    model.classifier = classifier
    #print(model.parameters)

    #можно сделать тут через exec подмену классифаера если имя последнего уровня отличается от classifier
    return model
    

def train_and_validate(args):
    trainloader, validloader, testloader, class_to_idx = define_transfroms(args)
    #print("class_to_idx in train and validate = ", class_to_idx)
        
    model = create_model(args)
    model.class_to_idx = class_to_idx
    #print("model.class_to_idx = ", model.class_to_idx)
    print ("gpu is available: ", torch.cuda.is_available())    
    print ("input parameter for enabling GPU: ", args[6])    
    device = torch.device("cuda:0" if torch.cuda.is_available() and args[6] == True else "cpu")
    print(f"device = {device}")
    model.to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args[3])
    train_losses, validation_losses, test_losses = [], [], []

    epochs = args[5]
    print ("Starting epochs...")
    with active_session():
        for e in range(epochs):
            #print(f"Starting epoch {e+1}...")
            running_loss = 0
            epoch_start_time = time.time()
            for ii, (inputs, labels) in enumerate(trainloader):
                #start = time.time()
                #print(f"ii: {ii}")
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model.forward(inputs)
                #print(outputs.shape)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                #print(f"Batch = {ii}; Time per batch: {(time.time() - start):.3f} seconds, loss {running_loss/len(trainloader)}")
            
                #if ii%30 == 0 or ii == 102:
                    #print(f"Batch = {ii}; loss {running_loss/len(trainloader):.5f}")
            else:
                #test_loss = 0
                validation_loss = 0
                #accuracy = 0 
                validation_accuracy = 0
                with torch.no_grad():
                    model.eval()
                    #print("Starting eval phase...")
                    #for ii, (images, labels) in enumerate(testloader):
                    for ii, (images, labels) in enumerate(validloader):
                        images, labels = images.to(device), labels.to(device)
                        log_ps = model(images)
                        #test_loss += criterion(log_ps, labels)
                        validation_loss += criterion(log_ps, labels)

                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim = 1)
                        equals = top_class == labels.view(*top_class.shape)
                        #accuracy += torch.mean(equals.type(torch.FloatTensor))
                        validation_accuracy += torch.mean(equals.type(torch.FloatTensor))
                    
                        #print(f"Eval batch={ii}, accuracy={accuracy}")
                train_losses.append(running_loss/len(trainloader))
                #test_losses.append(test_loss/len(testloader))
                validation_losses.append(validation_loss/len(validloader))
                model.train()
                save_load_model.save_the_checkpoint(model, args)

                #print(f"test loss: {test_loss/len(testloader):7.4f}, test accuracy: {accuracy/len(testloader):7.4f}")
                print(f"Epoch: {e+1}/{epochs}, training loss: {running_loss/len(trainloader):7.4f}, validation loss: {validation_loss/len(validloader):7.4f}, validation accuracy: {validation_accuracy/len(validloader):7.4f}, duration: {(time.time() - epoch_start_time):.3f} sec")        
    # TODO: Save the checkpoint 
    #print(image_datasets['train'].class_to_idx)
    #save_load_model.save_the_checkpoint(model, args)
          

                        #print(f"Device = {device}; Time per batch: {(time.time() - start):.3f} seconds    

def test_the_network(args):
    print("Entering testing function...")
    trainloader, validloader, testloader, class_to_idx = define_transfroms(args)
    #model = import_the_checkpoint.load_checkpoint_and_rebuild_the_model()            
    model = save_load_model.load_checkpoint_and_rebuild_the_model(args)            
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    model.to(device)
    test_loss = 0
    test_accuracy = 0
    criterion = nn.NLLLoss()
    test_losses = []
    with torch.no_grad():
        model.eval()
        #print("Starting eval phase...")
        for ii, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            log_ps = model(images)
            #test_loss += criterion(log_ps, labels)
            test_loss += criterion(log_ps, labels)

            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim = 1)
            equals = top_class == labels.view(*top_class.shape)
            #accuracy += torch.mean(equals.type(torch.FloatTensor))
            test_accuracy += torch.mean(equals.type(torch.FloatTensor))
                   
            print(f"Test batch={ii}, accuracy={test_accuracy/(ii+1)}")
       
        #print(f"test loss: {test_loss/len(testloader):7.4f}, test accuracy: {accuracy/len(testloader):7.4f}")
        print(f"Test loss: {test_loss/len(testloader):7.4f}, test accuracy: {test_accuracy/len(testloader):7.4f}")
    print("Exiting testing function...")


                    
if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
    args = get_input_args()
    train_and_validate(args)   
