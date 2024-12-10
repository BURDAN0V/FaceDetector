import numpy as np
import cv2
from imutils.video import FPS

class Detector:
    def __init__(self, use_cuda=False):
        self.faceModel = cv2.dnn.readNetFromCaffe("models/res10_300x300_ssd_iter_140000.prototxt",
        caffeModel="models/res10_300x300_ssd_iter_140000.caffemodel")

        if use_cuda:
            self.faceModel.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.faceModel.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    def processWebcam(self):
        cap = cv2.VideoCapture(0)  # Открываем веб-камеру (0 - первая подключенная камера)

        # Устанавливаем разрешение на HD (1280x720)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        #cap.set(cv2.flip)

        if not cap.isOpened():
            print("Error opening webcam...")
            return

        fps = FPS().start()

        while True:
            ret, self.img = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            (self.height, self.width) = self.img.shape[:2]

            self.processFrame()
            cv2.imshow("Webcam Output", self.img)  # Вывод кадра без изменения размера

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):  # Нажмите "q" для выхода
                break

            fps.update()

        fps.stop()
        print("Elapsed time: {:.2f}".format(fps.elapsed()))
        print("FPS: {:.2f}".format(fps.fps()))

        cap.release()
        cv2.destroyAllWindows()

    def processFrame(self):
        # Блоб для нейросети, но не изменяем само изображение
        blob = cv2.dnn.blobFromImage(self.img, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)

        self.faceModel.setInput(blob)
        predictions = self.faceModel.forward()

        # Прямоугольники рисуем на оригинальном изображении
        for i in range(0, predictions.shape[2]):
            if predictions[0, 0, i, 2] > 0.5:
                bbox = predictions[0, 0, i, 3:7] * np.array([self.width, self.height, self.width, self.height])
                (xmin, ymin, xmax, ymax) = bbox.astype("int")

                # Рисуем рамку на оригинальном изображении
                cv2.rectangle(self.img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

# Пример использования
detector = Detector(use_cuda=True)
detector.processWebcam()
