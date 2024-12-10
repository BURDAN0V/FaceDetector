from Detector import *

detector = Detector(use_cuda=False)

#---------Web-Cam---------------------------
detector.processWebcam()


#---------Images----------------------------
#detector.processImage("Images\Alena.jpg")


#---------Videos----------------------------
#detector.processVideo("Images\Diana_Cropped_29.mp4")
