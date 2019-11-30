# import socket programming library 
import socket
import pickle
import cv2

# import thread module
import _thread


class ImageData:
    img_name = "" # please define
    img_bytestream = [] # original image bye stream 
    aligned_face_bytestream = [] # aligned faces bytestream array in order from big faces to small faces
    box_posx = [] # x-coordinate array of extracting faces from original image 
    box_posy = [] # y- coordinate array of extracting faces from original image 
    box_w = [] # width array of extracting faces from original image
    box_h = [] # height array of extracting faces from original image


# thread fuction
def threaded(c):
    fragments = []
    while True:
         # data received from client
        data = c.recv(4096)

        if not data:
            break

        fragments.append(data)
    print('receive')

    arr = b''.join(fragments)

    # All received information
    data_receive = pickle.loads(arr)
    # img_bytestream = pickle.loads(data_receive.img_bytestream)
    aligned_face_bytestream = pickle.loads(data_receive.aligned_face_bytestream)
    box_posx = pickle.loads(data_receive.box_posx)
    box_posy = pickle.loads(data_receive.box_posy)
    box_w = pickle.loads(data_receive.box_w)
    box_h = pickle.loads(data_receive.box_h)
    
    print("Number of faces:", len(aligned_face_bytestream))
    # print("image_bytestream:", img_bytestream)
    print("box_posx:", box_posx)
    print("box_posy:", box_posy)
    print("box_w:", box_w)
    print("box_h:", box_h)
    c.close()


def Main():
    host = ""

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
        _thread.start_new_thread(threaded, (c,))
    s.close()


if __name__ == '__main__':
    Main()
