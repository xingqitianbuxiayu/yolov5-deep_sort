'''
测试视频检测+追踪效果，并生成result.mp4视频
'''
from objdetector import Detector
import imutils
import cv2

VIDEO_PATH = './test_data/test_person.mp4'
# VIDEO_PATH = './test_data/middoor.mp4'
# three_person_out_result.mp4
RESULT_PATH='.'+VIDEO_PATH.split('.')[-2]+'_deepsort_result.'+VIDEO_PATH.split('.')[-1]
# RESULT_PATH='.'+VIDEO_PATH.split('.')[-2]+'_yolov5l6_v2_10_result.'+VIDEO_PATH.split('.')[-1]

def main():

    func_status = {}
    func_status['headpose'] = None
    
    name = 'demo'

    det = Detector()
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = int(cap.get(5))
    print('fps:', fps)
    t = int(1000/fps)

    size = None
    videoWriter = None

    while True:

        # try:
        _, im = cap.read()
        if im is None:
            break
        
        result = det.feedCap(im, func_status)
        result = result['frame']
        # 一个是需要修改大小的图像，一个是weight或者是height，是根据的长宽比一起修改的
        # result = imutils.resize(result, height=500)
        if videoWriter is None:
            fourcc = cv2.VideoWriter_fourcc(
                'm', 'p', '4', 'v')  # opencv3.0
            videoWriter = cv2.VideoWriter(
                RESULT_PATH, fourcc, fps, (result.shape[1], result.shape[0]))

        videoWriter.write(result)
        cv2.imshow(name, result)
        cv2.waitKey(t)

        if cv2.getWindowProperty(name, cv2.WND_PROP_AUTOSIZE) < 1:
            # 点x退出
            break

    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    
    main()