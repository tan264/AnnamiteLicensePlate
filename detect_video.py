import time
import cv2
from detect_image import detect_image
import utils

def detect_video(vid, kNearest):
    video_width  = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)) 
    video_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))  
    video_fps = int(vid.get(cv2.CAP_PROP_FPS))
    prev_frame_time = 0
    new_frame_time = 0
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('results/output.mp4',fourcc, video_fps, (video_width, video_height))
    ret = True
    while(ret):
        ret, frame = vid.read()

        # Bắt từng frame, coi nó như là một ảnh và truyền vào xử lý ánh
        detect_image(frame, kNearest)
        
        out.write(frame)

        # Tính toán FPS
        # new_frame_time = time.time()
        # fps = 1/(new_frame_time-prev_frame_time)
        # prev_frame_time = new_frame_time
        # fps = int(fps)
        # cv2.putText(frame, str(fps), (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
        frame_copy = cv2.resize(frame, None, fx=0.5, fy=0.5)
        cv2.imshow('Number Plate Recognition', frame_copy)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video = cv2.VideoCapture("videos/demo3.mp4")
    kNearest = utils.initModel()
    detect_video(video, kNearest)