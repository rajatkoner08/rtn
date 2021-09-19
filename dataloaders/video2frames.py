import cv2
import os
from pathlib import Path
from tqdm import tqdm
from config import VIDEO_DATA_DIR
#get all the files in the directory
allFiles = os.listdir(VIDEO_DATA_DIR)
if len(allFiles)>0:
    for file in tqdm(allFiles):
        #create folder to store frames
        FRAME_DATA_DIR = os.path.join(Path(VIDEO_DATA_DIR).parent,"vidVRD-frames", os.path.splitext(file)[0])
        if not os.path.exists(FRAME_DATA_DIR):
            os.makedirs(FRAME_DATA_DIR)
        vidcap = cv2.VideoCapture(os.path.join(VIDEO_DATA_DIR, file))
        success,image = vidcap.read()
        count = 0
        while success:
              cv2.imwrite(FRAME_DATA_DIR+"/%04d.jpg" % count, image)     # save frame as JPEG file
              success,image = vidcap.read()
              count += 1
        #print("Completed file",file)
print("Sucessfully completed parsing frames")