import json
import numpy as np
import matplotlib.pyplot as plt

#def view_classify(ps_and_classes, display = True):

def get_key(dictionary, val_to_find): 
    for key, value in dictionary.items(): 
         if val_to_find == value: 
            return key 
    return "key doesn't exist"

def get_topk_labels(args_predict, ps_and_classes, display = True):
    topk5_labels = []
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        #print(type(cat_to_name))
        #print(cat_to_name)
    
    #print("ps_and_classes in view_classify = ", ps_and_classes)
    #print("ps_and_classes[0][0] = ", ps_and_classes[0][0])
    #print("ps_and_classes[0][1] = ", ps_and_classes[0][1])
    numpy_probs = ps_and_classes[0][0].cpu().data.numpy()
    numpy_classes = ps_and_classes[0][1].cpu().data.numpy()
    #print("numpy_probs = ", type(numpy_probs), numpy_probs)
    #print("numpy_classes:", type(numpy_classes), numpy_classes, numpy_classes.shape, numpy_classes[0])
    #numpy_classes: <class 'numpy.ndarray'> [[ 7 38 77  2 60]] (1, 5) [ 7 38 77  2 60]
    with open('class_to_idx.json', 'r') as f:
        class_to_idx = json.load(f)
        #print("class_to_idx type", type(class_to_idx))
        #class_to_idx type <class 'dict'>
        #i= 7
        #new_i= None
    for i in numpy_classes[0]:
        new_new_i = get_key(class_to_idx, i)
        #print (f"i={i}, new_new_i={new_new_i}")
        topk5_labels.append(cat_to_name[str(new_new_i)])
    #print("topk5_labels = ", topk5_labels)
    return topk5_labels, numpy_probs
    #labels = probs[1].data.numpy().squeeze()
    #print(labels[2], labels, type(labels))
    
def display_bar_histogram(numpy_probs, topk5_labels):
    #print ("this is what display_bar_histogram receives")
    #print ("numpy_probs", type(numpy_probs), numpy_probs.shape, numpy_probs)
    fig, ax2 = plt.subplots( ncols=1)
    #ax2.barh(np.arange(5), ps)
    numpy_probs = numpy_probs.squeeze()
    ax2.barh(np.arange(len(numpy_probs)), numpy_probs)
    ax2.set_yticks(np.arange(5))
    ax2.set_yticklabels(topk5_labels, size='small');
    ax2.set_title('Probability')
    ax2.set_xlim(0, 1.1)
        
