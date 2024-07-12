import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import tracker
from detector import Detector

if __name__ == '__main__':
    # 初始化进入和离开计数
    count_person = 0
    count_vehicle = 0

    # 初始化计数列表
    person_per_frame = []
    vehicle_per_frame = []
    person_in = [0]
    vehicle_in = [0]

    font_draw_number = cv2.FONT_HERSHEY_SIMPLEX
    draw_text_position = (int(960 * 0.01), int(540 * 0.05))

    # 初始化检测器
    detector = Detector()

    # 初始化视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (960, 540))

    # 打开视频
    # capture = cv2.VideoCapture('./video/IMG_6481.mov')
    capture = cv2.VideoCapture('./video/test.mp4')
    # 用于跟踪已经统计过的目标ID的集合
    counted_ids = set()

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    for _ in tqdm(range(total_frames)):
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
            vehicle_in.append(vehicle_in[-1])
            person_in.append(person_in[-1])

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
                vehicle_in[-1] += 1
            elif label == 'vehicle':
                count_vehicle += 1
                person_in[-1] += 1

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
        text_draw = f'Person: {count_person}, Vehicle: {count_vehicle}.'
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

    # 生成每帧人车数折线图
    plt.figure(figsize=(10, 5))
    plt.plot(person_per_frame, label='Person Count', color='blue')
    plt.plot(vehicle_per_frame, label='Vehicle Count', color='cyan')
    plt.xlabel('Frame')
    plt.ylabel('Counts per Frame')
    plt.title('Counts per Frame Over Time for Person and Vehicle')
    plt.legend()
    plt.savefig('frame_line_chart.png')
    plt.show()

    # 生成每帧进入人车数折线图
    plt.figure(figsize=(10, 5))
    plt.plot(person_in, label='Person In', color='blue')
    plt.plot(vehicle_in, label='Vehicle In', color='cyan')
    plt.xlabel('Frame')
    plt.ylabel('Counts')
    plt.title('Counts Over Time for Person and Vehicle')
    plt.legend()
    plt.savefig('in_line_chart.png')
    plt.show()
