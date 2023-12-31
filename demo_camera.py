import cv2
import copy
import numpy as np
import torch
from src import model
from src import util
from src.body import Body
from src.hand import Hand
from threading import Thread
body_estimation = Body('model/body_pose_model.pth')
hand_estimation = Hand('model/hand_pose_model.pth')
print(f"Torch device: {torch.cuda.get_device_name()}")
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
# Reduce image size for faster processing
width, height = 320, 240
cap.set(3, width)
cap.set(4, height)
def process_frame(oriImg):
    candidate, subset = body_estimation(oriImg)
    canvas = copy.deepcopy(oriImg)
    canvas = util.draw_bodypose(canvas, candidate, subset)
    hands_list = util.handDetect(candidate, subset, oriImg)
    all_hand_peaks = []
    for x, y, w, is_left in hands_list:
        peaks = hand_estimation(oriImg[y:y+w, x:x+w, :])
        peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
        peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
        all_hand_peaks.append(peaks)
    canvas = util.draw_handpose(canvas, all_hand_peaks)
    cv2.imshow('demo', canvas)
def capture_frames():
    while True:
        ret, oriImg = cap.read()
        process_frame(oriImg)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Start the capture_frames thread
capture_thread = Thread(target=capture_frames)
capture_thread.start()

# Wait for the capture_thread to finish
capture_thread.join()

cap.release()
cv2.destroyAllWindows()