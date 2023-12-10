import time
import cv2
from detect_image import detect_image
import utils

def detect_video(vid, kNearest):
    prev_frame_time = 0
    new_frame_time = 0

    ret = True
    while(ret):
        ret, frame = vid.read()

        # Moi mot frame coi la 1 anh va xu ly nhu xy ly anh
        detect_image(frame, kNearest)
        
        # Tinh toan fps
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        cv2.putText(frame, str(fps), (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
        frame_copy = cv2.resize(frame, None, fx=0.5, fy=0.5)
        cv2.imshow('Number Plate Recognition', frame_copy)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video = cv2.VideoCapture("videos/demo2.mp4")
    kNearest = utils.initModel()
    detect_video(video, kNearest)