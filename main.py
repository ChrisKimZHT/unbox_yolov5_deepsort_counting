import argparse
import datetime
import os

import cv2
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm

import tracker
from detector import Detector


def main(video_input: str, output_path: str, headless: bool = False):
    # 初始化输出文件夹
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 初始化计数
    count_person = 0
    count_vehicle = 0

    # 初始化计数列表
    person_per_frame = [0]
    vehicle_per_frame = [0]
    person_in = [0]
    vehicle_in = [0]

    # 打开视频
    capture = cv2.VideoCapture(video_input)

    # 用于跟踪已经统计过的目标ID的集合
    counted_ids = set()

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = capture.get(cv2.CAP_PROP_FPS)

    # 初始化视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.join(output_path, 'result_video.mp4'),
                          fourcc, original_fps, (960, 540))

    # 初始化检测器
    detector = Detector()

    for _ in tqdm(range(total_frames)):
        vehicle_in.append(vehicle_in[-1])
        person_in.append(person_in[-1])

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
                count_person += 1
                person_in[-1] += 1
            elif label == 'vehicle':
                count_vehicle += 1
                vehicle_in[-1] += 1

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
        current_count_person = sum(
            1 for bbox in list_bboxs if bbox[4] == 'person')
        current_count_vehicle = sum(
            1 for bbox in list_bboxs if bbox[4] == 'vehicle')

        # 收集当前帧上下行目标数量
        person_per_frame.append(current_count_person)
        vehicle_per_frame.append(current_count_vehicle)

        # 绘制统计信息到帧上
        font_draw_number = cv2.FONT_HERSHEY_SIMPLEX
        draw_text_position = (int(960 * 0.01), int(540 * 0.05))
        text_draw = f'Person: {count_person}, Vehicle: {count_vehicle}.'
        output_image_frame = cv2.putText(img=im, text=text_draw, org=draw_text_position,
                                         fontFace=font_draw_number, fontScale=1,
                                         color=(255, 255, 255), thickness=2)

        # 显示结果帧
        if not headless:
            cv2.imshow('demo', output_image_frame)

        # 写入结果帧
        if out.isOpened():
            out.write(output_image_frame)

        cv2.waitKey(1)

    capture.release()
    cv2.destroyAllWindows()

    # 利用帧数生成时间横坐标
    time_per_frame = [i / original_fps for i in range(total_frames + 1)]

    # 生成每帧人车数折线图
    plt.figure(figsize=(16, 9))
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(time_per_frame, person_per_frame,
             label='Person Count', color='blue')
    plt.plot(time_per_frame, vehicle_per_frame,
             label='Vehicle Count', color='cyan')
    plt.xlabel('Time (s)')
    plt.ylabel('Counts per Frame')
    plt.title('Counts per Frame Over Time for Person and Vehicle')
    plt.legend()
    plt.savefig(os.path.join(output_path, 'per_frame_line_chart.png'))
    if not headless:
        plt.show()

    # 生成每帧进入人车数折线图
    plt.figure(figsize=(16, 9))
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(time_per_frame, person_in, label='Person In', color='blue')
    plt.plot(time_per_frame, vehicle_in, label='Vehicle In', color='cyan')
    plt.xlabel('Time (s)')
    plt.ylabel('Counts')
    plt.title('Counts Over Time for Person and Vehicle')
    plt.legend()
    plt.savefig(os.path.join(output_path, 'in_line_chart.png'))
    if not headless:
        plt.show()

    # 生成人车进入的变化情况
    plt.figure(figsize=(16, 9))
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(time_per_frame, [0] + [person_in[i] - person_in[i - 1]
             for i in range(1, len(person_in))], label='Person In', color='blue')
    plt.plot(time_per_frame, [0] + [vehicle_in[i] - vehicle_in[i - 1]
             for i in range(1, len(vehicle_in))], label='Vehicle In', color='cyan')
    plt.xlabel('Time (s)')
    plt.ylabel('Counts')
    plt.title('Counts Over Time for Person and Vehicle')
    plt.legend()
    plt.savefig(os.path.join(output_path, 'in_change_line_chart.png'))
    if not headless:
        plt.show()


if __name__ == '__main__':
    default_output_path = f'outputs/{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}'

    paser = argparse.ArgumentParser()
    paser.add_argument('--input_video', type=str, default='video/test.mp4')
    paser.add_argument('--output_path', type=str, default=default_output_path)
    paser.add_argument('--headless', type=bool, default=False)
    args = paser.parse_args()

    main(args.input_video, args.output_path, args.headless)
