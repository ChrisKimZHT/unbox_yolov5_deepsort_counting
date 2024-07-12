import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from detector import Detector
import tracker

if __name__ == '__main__':
    # 初始化进入和离开计数
    down_count_person = 0
    up_count_person = 0
    down_count_vehicle = 0
    up_count_vehicle = 0

    # 初始化计数列表
    frame_up_counts_person = []
    frame_down_counts_person = []
    frame_up_counts_vehicle = []
    frame_down_counts_vehicle = []

    # 初始化时间列表
    time_stamps = []
    start_time = time.time()

    font_draw_number = cv2.FONT_HERSHEY_SIMPLEX
    draw_text_position = (int(960 * 0.01), int(540 * 0.05))

    # 初始化检测器
    detector = Detector()

    # 初始化视频写入对象
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter('output_video.avi', fourcc, 20.0, (960, 540))

    # 打开视频
    # capture = cv2.VideoCapture('./video/test.mp4')
    capture = cv2.VideoCapture('./video/IMG.mov')
    # 用于跟踪已经统计过的目标ID的集合
    counted_ids = set()

    while True:
        # 读取每帧图片
        ret, im = capture.read()
        if not ret:
            break

        # 缩小尺寸，1920x1080->960x540
        im = cv2.resize(im, (960, 540))
        list_bboxs = []


        # 检测目标
        bboxes = detector.detect(im)

        # 更新目标追踪信息
        bboxes2draw = tracker.update(bboxes, im)

        # 统计人流和车流数量
        for x1, y1, x2, y2, label, track_id in bboxes2draw:
            # 如果跟踪ID已经统计过，则跳过
            if track_id in counted_ids:
                continue

            # 绘制边界框和跟踪ID
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(im, f'Track ID: {track_id}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 统计人流和车流
            if label == 'person':
                if y1 < 540 / 2:
                    up_count_person += 1
                else:
                    down_count_person += 1
            elif label == 'vehicle':
                if y1 < 540 / 2:
                    up_count_vehicle += 1
                else:
                    down_count_vehicle += 1

            # 将已统计过的跟踪ID添加到集合中
            counted_ids.add(track_id)

        # 如果画面中有bbox
        if bboxes:
            list_bboxs = tracker.update(bboxes, im)
            # 画框
            output_image_frame = tracker.draw_bboxes(
                im, list_bboxs, line_thickness=None)
        else:
            # 如果画面中没有bbox
            output_image_frame = im
        # 当前帧上下行目标数量
        current_up_count_person = sum(
            1 for bbox in list_bboxs if bbox[1] < 540 / 2 and bbox[4] == 'person')  # 上半部分人数量
        current_down_count_person = sum(
            1 for bbox in list_bboxs if bbox[1] >= 540 / 2 and bbox[4] == 'person')  # 下半部分人数量
        current_up_count_vehicle = sum(
            1 for bbox in list_bboxs if bbox[1] < 540 / 2 and bbox[4] == 'vehicle')  # 上半部分车数量
        current_down_count_vehicle = sum(
            1 for bbox in list_bboxs if bbox[1] >= 540 / 2 and bbox[4] == 'vehicle')  # 下半部分车数量



        # 收集当前时间和帧计数
        current_time = time.time() - start_time
        time_stamps.append(current_time)
        frame_up_counts_person.append(current_up_count_person)
        frame_down_counts_person.append(current_down_count_person)
        frame_up_counts_vehicle.append(current_up_count_vehicle)
        frame_down_counts_vehicle.append(current_down_count_vehicle)


        # 绘制统计信息到帧上
        text_draw = f'DOWN: Person {down_count_person} , Vehicle {down_count_vehicle} | UP: Person {up_count_person} , Vehicle {up_count_vehicle}'
        output_image_frame = cv2.putText(img=im, text=text_draw,
                                         org=draw_text_position,
                                         fontFace=font_draw_number,
                                         fontScale=1, color=(255, 255, 255), thickness=2)

        # 显示结果帧
        cv2.imshow('demo', output_image_frame)
        if out.isOpened():  # 检查视频写入对象是否成功打开
            out.write(output_image_frame)
        cv2.waitKey(1)

    capture.release()
    cv2.destroyAllWindows()

    # 生成柱状图
    plt.figure(figsize=(10, 5))
    plt.bar(['Up Person', 'Down Person', 'Up Vehicle', 'Down Vehicle'],
            [up_count_person, down_count_person,
             up_count_vehicle, down_count_vehicle],
            color=['blue', 'cyan', 'green', 'yellow'])
    plt.title('Total Up and Down Counts for Person and Vehicle')
    plt.savefig('total_bar_chart.png')
    plt.show()

    # 生成折线图
    plt.figure(figsize=(10, 5))
    plt.plot(time_stamps, frame_up_counts_person,
             label='Up Person per Frame', color='blue')
    plt.plot(time_stamps, frame_down_counts_person,
             label='Down Person per Frame', color='cyan')
    plt.plot(time_stamps, frame_up_counts_vehicle,
             label='Up Vehicle per Frame', color='green')
    plt.plot(time_stamps, frame_down_counts_vehicle,
             label='Down Vehicle per Frame', color='yellow')
    plt.xlabel('Time (s)')
    plt.ylabel('Counts per Frame')
    plt.title('Counts per Frame Over Time for Person and Vehicle')
    plt.legend()
    plt.savefig('frame_line_chart.png')
    plt.show()
