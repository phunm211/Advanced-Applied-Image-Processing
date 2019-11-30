# pip install opencv-python
import cv2

# pip install imutils
import imutils
from imutils.face_utils import FaceAligner
from imutils.video import FPS
# from imutils.video import VideoStream, FileVideoStream

# pip install dlib or conda install -c conda-forge dlib
import dlib

import numpy as np
import time
import argparse
import re
import socket
import pickle


class ImageData:
    img_name = "" # please define
    img_bytestream = [] # original image bye stream 
    aligned_face_bytestream = [] # aligned faces bytestream array in order from big faces to small faces
    box_posx = [] # x-coordinate array of extracting faces from original image 
    box_posy = [] # y- coordinate array of extracting faces from original image 
    box_w = [] # width array of extracting faces from original image
    box_h = [] # height array of extracting faces from original image


def read_video():
    camera = cv2.VideoCapture("face_video.mov")
    # camera = FileVideoStream(args["video"]).start()
    time.sleep(2.0)
    read_frame(camera, False)


def cam():
    camera = cv2.VideoCapture(0)
    # camera = VideoStream(src=0).start()
    # camera = VideoStream(usePiCamera=True).start()
    time.sleep(2.0)
    read_frame(camera, False)


def convert_img_to_bytestream(img):
    is_success, im_buf_arr = cv2.imencode(".jpg", img)
    if is_success:
        return im_buf_arr.tobytes()
    else: 
        return None

def detect_faces(image, detections, faceAlignment, confidence):
    (h, w) = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    box_posx = []
    box_posy = []
    box_w = []
    box_h = []
    aligned_faces = []

    for i in range(0, detections.shape[2]):
        guarantee = detections[0, 0, i, 2]
        if guarantee > confidence:
            box = np.abs(detections[0, 0, i, 3:7] * np.array([w, h, w, h]))
            (startX, startY, endX, endY) = box.astype("int")

            box_posx.append(startX)
            box_posy.append(startY)
            box_w.append(endX - startX)
            box_h.append(endY - startY)

            faceROI = image[startY : endY, startX : endX]
            (fH, fW) = faceROI.shape[:2]

            # if the face is too small then skipping it
            if fW < 20 or fH < 20:
                continue
            
            # extract aligned face, if cannot then keep the original face
            rect = dlib.rectangle(startX, startY, endX, endY)
            faceAligned = faceAlignment.align(image, gray, rect)
            aligned_faces.append(convert_img_to_bytestream(faceAligned))

            # drawing bounding box
            text = "{:.2f}%".format(guarantee * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    # Sort lists by face height
    box_posx = sortByOrder(box_posx, box_h)
    box_posy = sortByOrder(box_posy, box_h)
    box_w = sortByOrder(box_w, box_h)
    aligned_faces = sortByOrder(aligned_faces, box_h)
    box_h = sorted(box_h, reverse=True)

    return box_posx, box_posy, box_w, box_h, aligned_faces


def sortByOrder(x, order):
    return [x for order, x in sorted(zip(order, x), reverse=True)]


def read_frame(camera, writeVideo):
    fps = FPS().start()
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
    predictor = dlib.shape_predictor(args["landmarks_predictor"])

     # Change width and height of aligned face if necessary
    faceAlignment = FaceAligner(predictor, desiredFaceWidth = 160, desiredFaceHeight = 160)
    confidence = args["confidence"]

    # connect to server on local computer
    # local host IP '127.0.0.1'
    host = '127.0.0.1'

    # Define the port on which you want to connect
    port = 12345

    while True:
        data_send = ImageData()
        aligned_face_bytestream = []
        box_posx_temp = []
        box_posy_temp = []
        box_w_temp = []
        box_h_temp = []
        
        (grabbed, frame) = camera.read()

        if args.get("video") and not grabbed:
            break

        frame = imutils.resize(frame, width = 800)
        # Do the face detection forward
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)
        net.setInput(blob)
        detections = net.forward()

        # Detect faces and their information
        box_posx, box_posy, box_w, box_h, aligned_faces = detect_faces(frame, detections, faceAlignment, confidence)
        box_posx_temp.extend(box_posx)
        box_posy_temp.extend(box_posy)
        box_w_temp.extend(box_w)
        box_h_temp.extend(box_h)
        aligned_face_bytestream.extend(aligned_faces)

        # Convert into bytestream
        data_send.img_bytestream = convert_img_to_bytestream(frame)
        data_send.aligned_face_bytestream = pickle.dumps(aligned_face_bytestream)
        data_send.box_posx = pickle.dumps(box_posx_temp)
        data_send.box_posy = pickle.dumps(box_posy_temp)
        data_send.box_w = pickle.dumps(box_w_temp)
        data_send.box_h = pickle.dumps(box_h_temp)
        data_string = pickle.dumps(data_send)

        # Send the information to server
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((host, port))
        s.sendall(data_string)
        s.close()
        
        # if writeVideo:
        #     out.write(frame)

        cv2.imshow("Frame", frame)
        fps.update()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    camera.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prototxt", default="models/deploy.prototxt")
    ap.add_argument("-m", "--model", default="models/res10_300x300_ssd_iter_140000.caffemodel")
    ap.add_argument('-l', '--landmarks-predictor', default="models/shape_predictor_5_face_landmarks.dat")
    ap.add_argument("-c", "--confidence", type=float, default=0.5)
    ap.add_argument("-v", "--video", help = "path to the (optional) video file")
    args = vars(ap.parse_args())

    # fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    # out = cv2.VideoWriter('output.avi', fourcc, 30.0, (800, 600))

    # read_video()
    cam()
