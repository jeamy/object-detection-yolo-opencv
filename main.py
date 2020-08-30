import sys
import time

import cv2
from flask import Flask, render_template, Response
from video import Camera
import argparse

# rtsp://admin:123456@192.168.8.50:554/h264Preview_01_main
# rtsp://admin:12345@192.168.8.51:554/live/main
# rtsp://prosmart:asgard69a%23ps@192.168.8.135:554/stream=0

parser = argparse.ArgumentParser()
parser.add_argument('--play_video', help="Tue/False", default=True)
# parser.add_argument('--video_path', help="Path of video file", default="/media/programming/testtvid/1.1515-video.mp4")
# parser.add_argument('--video_path', help="Path of video file", default="/media/programming/testtvid/2481-video.mp4")
parser.add_argument('--video_path', help="Path of video file", default="rtsp://prosmart:asgard69@192.168.8.135:554/")
parser.add_argument('--verbose', help="To print statements", default=True)
parser.add_argument('--scale', help="scale vid in percent", default=100)
parser.add_argument('--max_height', help="VVideo max height in pixel", default=900)
parser.add_argument('--weights', help="Yolo weights", default="./weigths/yolov4.weights")
parser.add_argument('--config', help="Yolo brain", default="./cfg/yolov4.cfg")
args = parser.parse_args()
delay = 5

app = Flask(__name__)


@app.route('/')
def index():
    # rendering webpage
    return render_template('index.html')


def gen(cam):
    while True:
        try:
            # get camera frame
            frame = cam.get_frame()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        except Exception as err:
            print("Read error:", err)
            print("Waiting " + str(delay) + " seconds ...")
            time.sleep(delay)
            continue
    time.sleep(250)


@app.route('/video_feed')
def video_feed():
    return Response(gen(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    video_play = args.play_video
    video_path = args.video_path
    try:
        while True:
            print("Opening stream " + video_path)
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                break
            print("Waiting for stream ...")
            time.sleep(delay)
            key = cv2.waitKey(0)
            if key == 27:
                sys.exit()
    except Exception as err:
        print("Connection error:", err)
        sys.exit()

    camera = Camera(cap, args)
    if args.verbose:
        print('Opening ' + video_path + " .... ")
        camera.start_video()
    # defining server ip address and port
    app.run(host='0.0.0.0', port='5000', debug=True)
