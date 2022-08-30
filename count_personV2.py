'''
检测+追踪+计数
'''
import numpy as np

import objtracker
from objdetector import Detector
import cv2

# VIDEO_PATH = './test_data/test_person.mp4'
VIDEO_PATH = './test_data/middoor.mp4'
# three_person_out_result.mp4
RESULT_PATH='.'+VIDEO_PATH.split('.')[-2]+'_yolov5l_v2_result.'+VIDEO_PATH.split('.')[-1]
# RESULT_PATH='.'+VIDEO_PATH.split('.')[-2]+'_yolov5l6_v2_10_result.'+VIDEO_PATH.split('.')[-1]


def drawLine(height,width):
    # 720×1280
    mask_image_temp = np.zeros((height, width), dtype=np.uint8)
    gaps=10
    thicknesses=50
    line_mid=(thicknesses+gaps)//2
    # 两条线的间隔要小,线宽要大
    # 填充第一个撞线polygon（蓝色）
    # list_pts_blue = [[204, 305], [227, 431], [605, 522], [1101, 464], [1900, 601], [1902, 495], [1125, 379], [604, 437],
    #                  [299, 375], [267, 289]]
    # 10×2的数组
    # ndarray_pts_blue = np.array(list_pts_blue, np.int32)
    # 填充任意形状的图形.可以用来绘制多边形
    # 720×1280
    # polygon_blue_value_1 = cv2.fillPoly(mask_image_temp, [ndarray_pts_blue], color=1)
    # 画线
    polygon_blue_value_1 = cv2.line(mask_image_temp, (0, height // 2 - line_mid), (width, height // 2 - line_mid), color=1,
                                    thickness=thicknesses)
    # 720×1280×1
    polygon_blue_value_1 = polygon_blue_value_1[:, :, np.newaxis]

    # 填充第二个撞线polygon（黄色）
    mask_image_temp = np.zeros((height, width), dtype=np.uint8)
    # list_pts_yellow = [[181, 305], [207, 442], [603, 544], [1107, 485], [1898, 625], [1893, 701], [1101, 568],
    #                    [594, 637], [118, 483], [109, 303]]
    # ndarray_pts_yellow = np.array(list_pts_yellow, np.int32)
    # polygon_yellow_value_2 = cv2.fillPoly(mask_image_temp, [ndarray_pts_yellow], color=2)
    # 画线
    polygon_yellow_value_2 = cv2.line(mask_image_temp, (0, height // 2 + line_mid), (width, height // 2 + line_mid), color=2,
                                    thickness=thicknesses)
    polygon_yellow_value_2 = polygon_yellow_value_2[:, :, np.newaxis]

    # 撞线检测用的mask，包含2个polygon，（值范围 0、1、2），供撞线计算使用
    # 在蓝线区域的值为1，黄线区域的值为2
    polygon_mask_blue_and_yellow = polygon_blue_value_1 + polygon_yellow_value_2

    # 蓝 色盘 b,g,r
    blue_color_plate = [255, 0, 0]
    # 蓝 polygon图片
    blue_image = np.array(polygon_blue_value_1 * blue_color_plate, np.uint8)

    # 黄 色盘
    yellow_color_plate = [0, 255, 255]
    # 黄 polygon图片
    yellow_image = np.array(polygon_yellow_value_2 * yellow_color_plate, np.uint8)

    # 彩色图片（值范围 0-255）
    color_polygons_image = blue_image + yellow_image
    return polygon_mask_blue_and_yellow,color_polygons_image

def judgeInorOut(list_bboxs,polygon_mask_blue_and_yellow):
    global list_overlapping_blue_polygon
    global list_overlapping_yellow_polygon
    global up_count
    global down_count
    if len(list_bboxs) > 0:
        # ----------------------判断撞线----------------------
        for item_bbox in list_bboxs:
            x1, y1, x2, y2, _, track_id = item_bbox
            # 撞线检测点，(x1，y1)，y方向偏移比例 0.0~1.0
            y1_offset = int(y1 + ((y2 - y1) * 0.6))
            # 撞线的点
            y = y1_offset
            x = x1
            if polygon_mask_blue_and_yellow[y, x] == 1:
                # 如果撞 蓝polygon
                if track_id not in list_overlapping_blue_polygon:
                    list_overlapping_blue_polygon.append(track_id)
                # 判断 黄polygon list里是否有此 track_id
                # 有此track_id，说明已经事先经过了黄线,则认为是 UP (上行)方向
                if track_id in list_overlapping_yellow_polygon:
                    # 上行+1
                    up_count += 1
                    print('out count:', up_count, ', out id:', list_overlapping_yellow_polygon)
                    # 删除 黄polygon list 中的此id
                    list_overlapping_yellow_polygon.remove(track_id)

            elif polygon_mask_blue_and_yellow[y, x] == 2:
                # 如果撞 黄polygon
                if track_id not in list_overlapping_yellow_polygon:
                    list_overlapping_yellow_polygon.append(track_id)
                # 判断 蓝polygon list 里是否有此 track_id
                # 有此 track_id，说明已经事先经过了蓝线,则认为是 DOWN（下行）方向
                if track_id in list_overlapping_blue_polygon:
                    # 下行+1
                    down_count += 1
                    print('in count:', down_count, ', in id:', list_overlapping_blue_polygon)
                    # 删除 蓝polygon list 中的此id
                    list_overlapping_blue_polygon.remove(track_id)
        # ----------------------清除无用id----------------------
        list_overlapping_all = list_overlapping_yellow_polygon + list_overlapping_blue_polygon
        for id1 in list_overlapping_all:
            is_found = False
            for _, _, _, _, _, bbox_id in list_bboxs:
                if bbox_id == id1:
                    is_found = True
            if not is_found:
                # 如果没找到，删除id
                if id1 in list_overlapping_yellow_polygon:
                    list_overlapping_yellow_polygon.remove(id1)

                if id1 in list_overlapping_blue_polygon:
                    list_overlapping_blue_polygon.remove(id1)
        list_overlapping_all.clear()
        # 清空list
        list_bboxs.clear()
    else:
        # 如果图像中没有任何的bbox，则清空list
        list_overlapping_blue_polygon.clear()
        list_overlapping_yellow_polygon.clear()


def main():
    # 帧间隔为frames-1
    frames=10
    # frames-1就比较，如果坐标y增大则表示是进，坐标y减小则表示是出
    inList=[]
    outList=[]
    # 第i号位置上存放第i帧中的所有框的信息，只保存frames个数据用于计算
    inforList=[]

    # ----------------打开视频，根据视频尺寸，填充供撞线计算使用的polygon------------------#
    capture = cv2.VideoCapture(VIDEO_PATH)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # video_width = int(capture.get(3))
    # video_height = int(capture.get(4))
    # fps = capture.get(5)
    fps = capture.get(cv2.CAP_PROP_FPS)
    print('fps:', fps)
    # cv.imshow()时每帧显示的时长
    # 需要注意的是, 保存视频的时候,每两帧之间需要等待一定时间,等待的时间为1000ms/视频帧率,要不然保存的视频会比原来播放的快.
    t = int(1000 / fps)
    # --------------------cv2.puttext函数的参数值-----------------------#

    # cv2.puttext函数的参数值
    font_draw_number = cv2.FONT_HERSHEY_SIMPLEX
    draw_text_postion = (int((width / 2) * 0.01), int((height / 2) * 0.05))
    videoWriter = None

    # --------------------两条彩色线，返回数组值-----------------------#
    # polygon_mask_blue_and_yellow,color_polygons_image=drawLine(height,width)
    # 缩小尺寸，1920x1080->960x540
    # color_polygons_image = cv2.resize(color_polygons_image, (width // 2, height // 2))
    # polygon_mask_blue_and_yellow = cv2.resize(polygon_mask_blue_and_yellow, (width // 2, height // 2))

    # 实例化yolov5检测器
    detector = Detector()
    # 表明当前是第几帧图片
    frame_id=0
    while True:
        # 读取每帧图片，返回是否成功标识(True,False)；img为读取的视频帧
        _, im = capture.read()
        if im is None:
            break

        # 缩小尺寸，1920x1080->960x540
        im = cv2.resize(im, (width // 2, height // 2))

        # list_bboxs = []
        # 返回检测到的图片和框
        _, bboxes = detector.detect(im)
        # 更新跟踪器
        output_image_frame, list_bboxs = objtracker.update(detector, im)
        # 添加两条检测线到原图片
        # output_image_frame = cv2.add(output_image_frame, color_polygons_image)
        # judgeInorOut(list_bboxs, polygon_mask_blue_and_yellow)
        dict_infor=0
        # list_bboxs表示当前图片帧中的所有框
        if len(list_bboxs) > 0:
            # 存储第i帧中所有框的id及框的y轴中心点坐标
            dict_infor={}
            # ----------------------判断撞线----------------------
            for item_bbox in list_bboxs:
                x1, y1, x2, y2, _, track_id = item_bbox
                # 撞线检测点，(x1，y1)，y方向偏移比例 0.0~1.0
                y1_offset = (y2+y1)/2
                # 撞线的点
                # track_id和y都为int类型
                dict_infor[track_id]=y1_offset
        else:
            # dict_infor=0
            print('第',frame_id,'帧中无框')

        # 间隔为frames-1，存储frames个数
        if len(inforList) < frames:
            inforList.append(dict_infor)
        else:#第frames+1张图片读到时，开始计算第frames张与第1张的差值
            frame_end = inforList[-1]
            for frame_start in inforList[0:-1]:
                # 如果frame_end=0，即无框，那么就不比较了
                if frame_end==0:
                    break
                # 如果frame_start=0，即无框，那么就比较下一个
                if frame_start==0:
                    continue
                for key,value in frame_end.items():
                    if key in frame_start.keys():
                        if value>frame_start[key]:
                            if key not in inList and key not in outList:
                                inList.append(key)
                            else:
                                pass
                        elif value<frame_start[key]:
                            if key not in outList and key not in inList:
                                outList.append(key)
                            else:
                                pass
                    else:
                        # 不加1的话表示帧序号从0开始
                        print('第',frame_id,'帧中有',key,'-id，而第',frame_id-frames+1+inforList[0:-1].index(frame_start),'帧中未检测到',key,'-id')
            inforList = inforList[1:]

        # 输出计数信息
        text_draw = 'In: ' + str(len(inList)) + \
                    ' , Out: ' + str(len(outList))
        output_image_frame = cv2.putText(img=output_image_frame, text=text_draw,
                                         org=draw_text_postion,
                                         fontFace=font_draw_number,
                                         fontScale=0.75, color=(0, 0, 255), thickness=2)
        # 将结果保存为mp4文件，宽度和高度为原来的一半
        if videoWriter is None:
            # YUV编码，后缀为.avi格式。会产生大文件。   fourcc = cv2.VideoWriter_fourcc('I','4','2','0')
            # MPEG-1编码类型，文件名后缀为.avi。    cv2.VideoWriter_fourcc('P', 'I', 'M', 'I')
            # MPEG-4编码，后缀为.avi格式
            fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
            # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            if VIDEO_PATH.split('.')[-1]=='mp4':

                # fourcc = cv2.VideoWriter_fourcc('x', '2', '6', '4')
                # MPEG-4编码，后缀为.mp4
                fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            videoWriter = cv2.VideoWriter(RESULT_PATH, fourcc, fps, (output_image_frame.shape[1], output_image_frame.shape[0]))

        videoWriter.write(output_image_frame)
        cv2.imshow('Counting Demo', output_image_frame)
        cv2.waitKey(1)

        if cv2.getWindowProperty('Counting Demo', cv2.WND_PROP_AUTOSIZE) < 1:
            # 点窗口右上角的x退出
            break
        frame_id+=1

    capture.release()
    videoWriter.release()
    cv2.destroyAllWindows()
    print('进的ids:',inList)
    print('出的ids:',outList)

if __name__ == '__main__':
    # 下行数量
    down_count = 0
    # 上行数量
    up_count = 0
    # list 与蓝色polygon重叠
    list_overlapping_blue_polygon = []
    # list 与黄色polygon重叠
    list_overlapping_yellow_polygon = []
    main()

