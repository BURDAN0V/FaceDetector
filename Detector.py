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


    def processImage(self, imgName):
        self.img = cv2.imread(imgName)
        (self.height, self.width) = self.img.shape[:2]

        # Вывод оригинального разрешения изображения
        print(f"Original resolution: {self.width}x{self.height}")

        self.processFrame()

        cv2.imshow("Output", self.img)  # Вывод изображения без изменения размера
        cv2.waitKey(0)

    def processVideo(self, videoName):
        cap = cv2.VideoCapture(videoName)
        if (cap.isOpened() == False):
            print("Error opening video...")
            return

        (success, self.img) = cap.read()
        (self.height, self.width) = self.img.shape[:2]

        # Вывод оригинального разрешения первого кадра видео
        print(f"Original video resolution: {self.width}x{self.height}")

        # Устанавливаем размер видео на FHD (1920x1080)
        target_width = 1920
        target_height = 1080

        fps = FPS().start()

        while success:
            # Изменяем размер кадра на FHD
            self.img = cv2.resize(self.img, (target_width, target_height))

            self.processFrame()
            cv2.imshow("Output", self.img)  # Вывод измененного размера кадра

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            fps.update()
            (success, self.img) = cap.read()

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
        target_width = 1920
        target_height = 1080
        for i in range(0, predictions.shape[2]):
            if predictions[0, 0, i, 2] > 0.5:
                bbox = predictions[0, 0, i, 3:7] * np.array([target_width, target_height, target_width, target_height])
                (xmin, ymin, xmax, ymax) = bbox.astype("int")

                # Рисуем рамку на оригинальном изображении
                cv2.rectangle(self.img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

# Пример использования
#detector = Detector(use_cuda=True)
#detector.processWebcam()
