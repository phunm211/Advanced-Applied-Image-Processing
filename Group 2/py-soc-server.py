# import socket programming library
import numpy as np
import socket
import pickle
import torch
from torchvision import datasets, transforms, models
from PIL import Image
import json
from glob import glob
import io
from pymongo import MongoClient
#from AGE_recognition.AGE_recognition import predict
#import cv2
# import thread module
from _thread import *
import threading

# cuda availiability check
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

path = 'aligned_faces/test/'

# class names
class_name = ['Cuong', 'Dat', 'Eldor', 'Jinhee', 'Joon Huyng', 'Minh', 'Phu', 'Abdullah']

# transforms
loader = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


# single test image input for testing
def image_loader(image_bytestream):
    """load image, returns cuda tensor"""
    image = Image.open(io.BytesIO(image_bytestream))
    image = loader(image).float()
    image = torch.tensor(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image

# load model on CPU mode
model = torch.load('final_model.pth', map_location=lambda storage, loc: storage)


# paths = 'aligned_faces/test/*.jpg'


# face classification
def face_recog(paths):
    model.eval()
    images_so_far = 0
    name_array = []
    prediction_score = []
    with torch.no_grad():
        for i, inputs in enumerate(paths):
            input = image_loader(inputs)
            # input = input.to(device)
            outputs = model(input)
            _, preds = torch.max(outputs, 1)
            sm = torch.nn.Softmax()
            prob = sm(outputs)
            prob = np.array(torch.topk(prob, k=1))
            # print(preds)
            for j in range(input.size()[0]):
                # print('Box index: ', images_so_far)
                # print('predicted: {}'.format(class_name[preds[j].cpu().numpy()]))
                name_array.append('{}'.format(class_name[preds[j].cpu().numpy()]))
                prediction_score.append(prob[0] * 100)
                images_so_far += 1
    return name_array, prediction_score


class ImageData:
    img_bytestream = 0 # original image bye stream
    aligned_face_bytestream = 0 # aligned faces bytestream array in order from big faces to small faces
    box_posx = 0 # x-coordinate array of extracting faces from original image
    box_posy = 0 # y- coordinate array of extracting faces from original image
    box_w = 0 # width array of extracting faces from original image
    box_h = 0 # height array of extracting faces from original image


# For MongoDB
client = MongoClient('mongodb://223.195.37.85:27017/')
db = client['machine-learning']
collection = db['ml-collection']


# thread fuction
def threaded(c):
    # Step 1: Receive information from client via TCP/IP Socket
    fragments = []
    while True:
        # data received from client
        data = c.recv(4096)
        if not data:
            break
        fragments.append(data)
    arr = b''.join(fragments)

    # All received information
    data_receive = pickle.loads(arr)
    img_bytestream = data_receive.img_bytestream
    box_bytestream = pickle.loads(data_receive.aligned_face_bytestream)
    box_posx = pickle.loads(data_receive.box_posx)
    box_posy = pickle.loads(data_receive.box_posy)
    box_w = pickle.loads(data_receive.box_w)
    box_h = pickle.loads(data_receive.box_h)
    img_numbox = len(box_bytestream)

    # Step 2: Preprocessing bounding box image
    # Step 3: Get name and other information of bounding box image based on trained model
    box_info = face_recog(box_bytestream)
    box_labelname = box_info[0]
    box_score = box_info[1]

    # Step 4: Generate a JSON form and send to Group 3 via HTTP
    image_dict = dict()
    # image_dict["_id"] = collection.count()
    image_dict["image_bytestream"] = img_bytestream
    annotation_list = []
    for i in range(img_numbox):
        box_dict = dict()
        box_dict["label"] = box_labelname[i]
        box_dict["score"] = box_score[i]
        bbox = dict()
        bbox["x"] = int(box_posx[i])
        bbox["y"] = int(box_posy[i])
        bbox["w"] = int(box_w[i])
        bbox["h"] = int(box_h[i])
        box_dict["bbox"] = bbox
        # nparr = np.fromstring(box_bytestream[i], np.uint8)
        # img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        annotation_list.append(box_dict)
        # annotation_list = predict(img, annotation_list)
    image_dict["annotation"] = annotation_list
    nparr = np.fromstring(img_bytestream, np.uint8)
    #img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    #ann = predict(img, image_dict["annotation"])

    collection.insert_one(image_dict)

    #image_json = json.dumps(image_dict, indent=4)
    print(image_dict)
    c.close()


def Main():
    host = "223.195.37.238"
    #host = "127.0.0.1"
    # reverse a port on your computer
    # in our case it is 12345 but it
    # can be anything
    port = 12345
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((host, port))
    print("socket binded to port", port)

    # put the socket into listening mode
    s.listen(5)
    print("socket is listening")

    # a forever loop until client wants to exit
    while True:
        # establish connection with client
        c, addr = s.accept()

        # lock acquired by client
        # print_lock.acquire()
        print('Connected to :', addr[0], ':', addr[1])

        # Start a new thread and return its identifier
        start_new_thread(threaded, (c,))
    s.close()


if __name__ == '__main__':
    Main()
