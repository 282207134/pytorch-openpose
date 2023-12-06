import os
import json
import argparse
import copy
import numpy as np
import cv2
from typing import NamedTuple
from tqdm import tqdm

from demo_video import ffprobe
from src import util
from src.body import Body
from src.hand import Hand
import imageio

# 初始化 OpenPose 模型
body_estimation = Body('model/body_pose_model.pth')
hand_estimation = Hand('model/hand_pose_model.pth')

def process_frame(frame, body=True, hands=True):
    canvas = copy.deepcopy(frame)
    if body:
        candidate, subset = body_estimation(frame)
        canvas = util.draw_bodypose(canvas, candidate, subset)
    if hands:
        hands_list = util.handDetect(candidate, subset, frame)
        all_hand_peaks = []
        for x, y, w, is_left in hands_list:
            peaks = hand_estimation(frame[y:y+w, x:x+w, :])
            peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
            peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
            all_hand_peaks.append(peaks)
        canvas = util.draw_handpose(canvas, all_hand_peaks)
    return canvas

def generate_output_video(video_file, output_file, input_fps, input_framesize, output_params):
    cap = cv2.VideoCapture(video_file)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = imageio.get_writer(output_file, fps=input_fps, quality=output_params.get("q", 5))

    for frame_num in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if frame is None:
            break

        posed_frame = process_frame(frame)

        writer.append_data(posed_frame)

        cv2.imshow('frame', posed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if frame_num == total_frames - 1:
            # 当视频完全处理时跳出循环
            break

    cap.release()
    writer.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a video annotating poses detected.")
    parser.add_argument('file', type=str, help='Video file location to process.')
    args = parser.parse_args()

    video_file = args.file

    # 获取视频文件信息
    ffprobe_result = ffprobe(video_file)
    info = json.loads(ffprobe_result.json)
    videoinfo = [i for i in info["streams"] if i["codec_type"] == "video"][0]
    input_fps = eval(videoinfo["avg_frame_rate"])
    print("Input FPS:", input_fps)
    input_framesize = (int(videoinfo["width"]), int(videoinfo["height"]))

    # 定义输出文件名
    postfix = info["format"]["format_name"].split(",")[0]

    # 定义不同参数的输出视频文件
    output_params_list = [
        {"ffmpeg_video_codec": "libx264", "ffmpeg_preset": "medium"},
        {"ffmpeg_video_codec": "libx264", "ffmpeg_preset": "ultrafast"},
        {"ffmpeg_video_codec": "libxvid", "ffmpeg_preset": "fast"},
        {"ffmpeg_video_codec": "libvpx-vp9"},
    ]

    for i, output_params in enumerate(output_params_list):
        output_file = f"{video_file}_output_{i}.{postfix}"
        generate_output_video(video_file, output_file, input_fps, input_framesize, output_params)
