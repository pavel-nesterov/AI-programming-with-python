import save_load_model
import json
import torch
from  random import randint 
import process_image as pima
import train
import argparse
import view_classify
import signal

from contextlib import contextmanager

import requests

def get_input_args():
    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument('--topk', type = int, default = 5, help = 'Number of top classes to be returned') 
        in_args = parser.parse_args()
        in_args_dict = vars(in_args)
        #predict_in_args_list = [v for v in in_args_dict.values()]
        print(in_args_dict)
        return in_args_dict
   # stuff only to run when not called via 'import' here
    else:
        in_args_dict = {'topk': 5}
        return(in_args_dict)

DELAY = INTERVAL = 4 * 60  # interval time in seconds
MIN_DELAY = MIN_INTERVAL = 2 * 60
KEEPALIVE_URL = "https://nebula.udacity.com/api/v1/remote/keep-alive"
TOKEN_URL = "http://metadata.google.internal/computeMetadata/v1/instance/attributes/keep_alive_token"
TOKEN_HEADERS = {"Metadata-Flavor":"Google"}


def _request_handler(headers):
    def _handler(signum, frame):
        requests.request("POST", KEEPALIVE_URL, headers=headers)
    return _handler


@contextmanager
def active_session(delay=DELAY, interval=INTERVAL):
    """
    Example:

    from workspace_utils import active session

    with active_session():
        # do long-running work here
    """
    token = requests.request("GET", TOKEN_URL, headers=TOKEN_HEADERS).text
    headers = {'Authorization': "STAR " + token}
    delay = max(delay, MIN_DELAY)
    interval = max(interval, MIN_INTERVAL)
    original_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _request_handler(headers))
        signal.setitimer(signal.ITIMER_REAL, delay, interval)
        yield
    finally:
        signal.signal(signal.SIGALRM, original_handler)
        signal.setitimer(signal.ITIMER_REAL, 0)


def keep_awake(iterable, delay=DELAY, interval=INTERVAL):
    """
    Example:

    from workspace_utils import keep_awake

    for i in keep_awake(range(5)):
        # do iteration with lots of work here
    """
    with active_session(delay, interval): yield from iterable


def predict(image_path, model, args, args_predict, topk=5):
    #print ("entering predict function...") 
    img_tensor = pima.process_image(image_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() and args[6] == True else "cpu")
    #print(f"device = {device}")
    model.to(device)
    img_tensor = img_tensor.to(device)
    img_tensor = img_tensor.unsqueeze_(0)
    log_ps = model(img_tensor)
    ps = torch.exp(log_ps)
    ps_topk = ps.topk(topk)
    
    probs_and_classes = [ps_topk]
    print ("outcome from predict.predict = ", probs_and_classes)
    return probs_and_classes


def define_image_path_for_inference():
    image_paths_100 = ["aipnd-project_original udacity folder/flowers/test/100/image_07896.jpg", 
               "aipnd-project_original udacity folder/flowers/test/100/image_07897.jpg", 
               "aipnd-project_original udacity folder/flowers/test/100/image_07899.jpg",
               "aipnd-project_original udacity folder/flowers/test/100/image_07902.jpg",
               "aipnd-project_original udacity folder/flowers/test/100/image_07926.jpg",
               "aipnd-project_original udacity folder/flowers/test/100/image_07936.jpg",
               "aipnd-project_original udacity folder/flowers/test/100/image_07938.jpg",
               "aipnd-project_original udacity folder/flowers/test/100/image_07939.jpg"]
    
    image_paths_13 = ["aipnd-project_original udacity folder/flowers/test/13/image_05745.jpg", 
               "aipnd-project_original udacity folder/flowers/test/13/image_05761.jpg", 
               "aipnd-project_original udacity folder/flowers/test/13/image_05767.jpg",
               "aipnd-project_original udacity folder/flowers/test/13/image_05769.jpg",
               "aipnd-project_original udacity folder/flowers/test/13/image_05775.jpg",
               "aipnd-project_original udacity folder/flowers/test/13/image_05787.jpg"]
    
    image_path_index = randint(0,len(image_paths_100)-1)
    image_path = image_paths_100[image_path_index]
    print("image path = ", image_path)
    return image_path

def prepare_and_run_prediction(image_path):
    args_train = train.get_input_args()
    args_predict = get_input_args()
    #model = import_the_checkpoint_ver2.load_checkpoint_and_rebuild_the_model(args)
    model = save_load_model.load_checkpoint_and_rebuild_the_model(args_train)
    with torch.no_grad():
        model.eval()
        #image_path = define_image_path_for_inference()
        probs_and_classes = predict(image_path, model, args_train, args_predict)
        #print("started topklabels ")
        topk_labels, numpy_probs = view_classify.get_topk_labels(args_predict, probs_and_classes, False)
        print("completed topklabels", topk_labels)
        return probs_and_classes, topk_labels, numpy_probs
    
                    
if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
    image_path = define_image_path_for_inference()
    prepare_and_run_prediction(image_path)
