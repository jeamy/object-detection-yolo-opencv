import cv2
import numpy as np


class Camera:

    def __init__(self, cap, args):
        self.cap = cap
        self.args = args
        self.model, self.classes, self.colors, self.output_layers = self.load_yolo()
        # defining face detector
        # self.face_cascade = cv2.CascadeClassifier("./haarcascades/haarcascade_frontalface_alt2.xml")
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.args.verbose:
            print('Frame count:', frame_count)
            print('Frame width:', height)
            print('Frame height:', int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            print('Frame rate:', self.cap.get(cv2.CAP_PROP_FPS))
            print("Scale: " + str(self.args.scale))

        if int(self.args.scale) == 100 and height != 0:
            self.args.scale = int(int(self.args.max_height) / height * 100)
            if self.args.verbose:
                print("Scale: " + str(self.args.scale))

    def __del__(self):
        # releasing camera
        self.cap.release()

    # Load yolo
    def load_yolo(self):
        net = cv2.dnn.readNet(self.args.weights, self.args.config)
        classes = []
        with open("coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]

        layers_names = net.getLayerNames()
        output_layers = [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        # colors = np.random.uniform(0, 255, size=(len(classes), 3))
        colors = np.random.uniform(0, 255, size=(len(classes), 3))
        return net, classes, colors, output_layers

    def detect_objects(self, img, net, output_layers):
        blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)
        self.forward = net.forward(output_layers)
        outputs = self.forward
        return blob, outputs

    def get_box_dimensions(self, outputs, height, width):
        boxes = []
        confs = []
        class_ids = []
        for output in outputs:
            for detect in output:
                scores = detect[5:]
                class_id = np.argmax(scores)
                conf = scores[class_id]
                if conf > float(self.args.confidence):
                    center_x = int(detect[0] * width)
                    center_y = int(detect[1] * height)
                    w = int(detect[2] * width)
                    h = int(detect[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confs.append(float(conf))
                    class_ids.append(class_id)
        return boxes, confs, class_ids

    def rescale_frame(self, frame, percent=75):
        width = int(frame.shape[1] * percent / 100)
        height = int(frame.shape[0] * percent / 100)
        dim = (width, height)
        return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    def draw_labels(self, boxes, confs, colors, class_ids, classes, img):
        indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_DUPLEX
        # override color
        color = (0, 255, 0)
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                # color = colors[i]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y - 5), font, 1, color, 2)
                cv2.putText(img, "w: " + str(w), (x, y + h + 35), font, 1, color, 2)
                cv2.putText(img, "h: " + str(h), (x, y + h + 70), font, 1, color, 2)
        img = self.rescale_frame(img, percent=int(self.args.scale))
        return img

    # cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
    def get_frame(self):
        _, frame = self.cap.read()
        height, width, channels = frame.shape
        blob, outputs = self.detect_objects(frame, self.model, self.output_layers)
        boxes, confs, class_ids = self.get_box_dimensions(outputs, height, width)
        img = self.draw_labels(boxes, confs, self.colors, class_ids, self.classes, frame)
        ret, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes()

    # for later use
    # def get_image(self):
    #     # extracting frames
    #     ret, frame = self.video.read()
    #     frame = cv2.resize(frame, None, fx=int(self.args.scale/100), fy=int(self.args.scale/100),
    #                        interpolation=cv2.INTER_AREA)
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)
    #     for (x, y, w, h) in face_rects:
    #         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #         break  # encode OpenCV raw frame to jpg and displaying it
    #     ret, jpeg = cv2.imencode('.jpg', frame)
    #     return jpeg.tobytes()
