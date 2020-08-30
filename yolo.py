import cv2
import numpy as np
import argparse
import time

# rtsp://admin:123456@192.168.8.50:554/h264Preview_01_main
# rtsp://admin:12345@192.168.8.51:554/live/main
# rtsp://prosmart:asgard69a%23ps@192.168.8.135/stream=0

parser = argparse.ArgumentParser()
parser.add_argument('--webcam', help="True/False", default=False)
parser.add_argument('--play_video', help="Tue/False", default=True)
parser.add_argument('--image', help="Tue/False", default=False)
# parser.add_argument('--video_path', help="Path of video file", default="/media/programming/testtvid/1.1515-video.mp4")
parser.add_argument('--video_path', help="Path of video file", default="/media/programming/testtvid/2481-video.mp4")
# parser.add_argument('--video_path', help="Path of video file",
#                    default="rtsp://admin:123456@192.168.8.50:554/h264Preview_01_main")
parser.add_argument('--image_path', help="Path of image to detect objects", default="Images/bicycle.jpg")
parser.add_argument('--verbose', help="To print statements", default=True)
parser.add_argument('--scale', help="scale vid in percent", default=100)
parser.add_argument('--max_height', help="Video max height in pixel", default=900)
args = parser.parse_args()


def detect_objects(img, net, output_layers):
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
    return blob, outputs


def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


# Load yolo
def load_yolo():
    net = cv2.dnn.readNet("./weigths/yolov4.weights", "./cfg/yolov4.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    layers_names = net.getLayerNames()
    output_layers = [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    # colors = np.random.uniform(0, 255, size=(len(classes), 3))
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return net, classes, colors, output_layers


def load_image(img_path):
    # image loading
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape
    return img, height, width, channels


def start_webcam():
    cap = cv2.VideoCapture(0)
    return cap


def display_blob(blob):
    """
        Three images each for RED, GREEN, BLUE channel
    """
    for b in blob:
        for n, imgb in enumerate(b):
            cv2.imshow(str(n), imgb)


def get_box_dimensions(outputs, height, width):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0.3:
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


def draw_labels(boxes, confs, colors, class_ids, classes, img):
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_DUPLEX
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
    img = rescale_frame(img, percent=int(args.scale))
    cv2.imshow("Image", img)


def image_detect(imgpath):
    model, classes, colors, output_layers = load_yolo()
    image, height, width, channels = load_image(imgpath)
    blob, outputs = detect_objects(image, model, output_layers)
    boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
    draw_labels(boxes, confs, colors, class_ids, classes, image)
    while True:
        key = cv2.waitKey(1)
        if key == 27:
            break


def webcam_detect():
    model, classes, colors, output_layers = load_yolo()
    cap = start_webcam()
    while True:
        _, frame = cap.read()
        height, width, channels = frame.shape
        blob, outputs = detect_objects(frame, model, output_layers)
        boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
        draw_labels(boxes, confs, colors, class_ids, classes, frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
        else:
            print(key)  # else print its value
    cap.release()


def start_video(videopath):
    model, classes, colors, output_layers = load_yolo()
    cap = cv2.VideoCapture(videopath)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('Frame count:', frame_count)
    heigth = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print('Frame width:', heigth)
    print('Frame height:', int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print('Frame rate:', cap.get(cv2.CAP_PROP_FPS))
    print("Scale: " + args.scale)

    if int(args.scale) == 100:
        args.scale = int(int(args.max_height) / heigth * 100)
        print("Scale: " + str(args.scale))

    # cap.set(cv2.CAP_PROP_POS_FRAMES, 100)

    while True:
        try:
            _, frame = cap.read()
            height, width, channels = frame.shape
            blob, outputs = detect_objects(frame, model, output_layers)
            boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
            draw_labels(boxes, confs, colors, class_ids, classes, frame)
        except Exception as err:
            print("Unexpected error:", err)
            continue

        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == 32:
            print(key)  # else print its value
            key = cv2.waitKey(-1)
    cap.release()


if __name__ == '__main__':
    webcam = args.webcam
    video_play = args.play_video
    image = args.image
    if webcam:
        if args.verbose:
            print('---- Starting Web Cam object detection ----')
        webcam_detect()
    if video_play:
        video_path = args.video_path
        if args.verbose:
            print('Opening ' + video_path + " .... ")
        start_video(video_path)
    if image:
        image_path = args.image_path
        if args.verbose:
            print("Opening " + image_path + " .... ")
        image_detect(image_path)

    cv2.destroyAllWindows()
