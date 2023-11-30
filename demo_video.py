import os
import json
import argparse
import copy
import numpy as np
import cv2
import subprocess
from typing import NamedTuple
from tqdm import tqdm
from src import util
from src.body import Body
from src.hand import Hand
import ffmpeg
import imageio

# 初始化 OpenPose 模型
body_estimation = Body('model/body_pose_model.pth')
hand_estimation = Hand('model/hand_pose_model.pth')

class Writer():
    def __init__(self, output_file, input_fps, input_framesize, output_params):
        if os.path.exists(output_file):
            os.remove(output_file)
        self.writer = imageio.get_writer(output_file, fps=input_fps, quality=output_params.get("q", 5))

    def __call__(self, frame):
        self.writer.append_data(frame)

    def close(self):
        self.writer.close()

class FFProbeResult(NamedTuple):
    return_code: int
    json: str
    error: str

def ffprobe(file_path) -> FFProbeResult:
    command_array = ["ffprobe",
                     "-v", "quiet",
                     "-print_format", "json",
                     "-show_format",
                     "-show_streams",
                     file_path]
    result = subprocess.run(command_array, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    return FFProbeResult(return_code=result.returncode,
                         json=result.stdout,
                         error=result.stderr)

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

    # 在这里添加颜色空间转换
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)  # 从BGR转换为RGB

    return canvas

def generate_output_video(video_file, output_file, input_fps, input_framesize, output_params):
    cap = cv2.VideoCapture(video_file)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = Writer(output_file, input_fps, input_framesize, output_params)

    for frame_num in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if frame is None:
            break

        # 在这里添加亮度和对比度调整
        frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)

        posed_frame = process_frame(frame)

        writer(posed_frame)

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
    parser = argparse.ArgumentParser(description="Process a video annotating poses detected.")
    parser.add_argument('file', type=str, help='Video file location to process.')
    args = parser.parse_args()
    video_file = args.file

    # 获取视频文件信息
    ffprobe_result = ffprobe(args.file)
    info = json.loads(ffprobe_result.json)
    videoinfo = [i for i in info["streams"] if i["codec_type"] == "video"][0]
    input_fps = eval(videoinfo["avg_frame_rate"])
    print("Input FPS:", input_fps)
    input_framesize = (int(videoinfo["width"]), int(videoinfo["height"]))
    output_params = {'q': 5}  # 调整输出参数，例如quality

    # 定义一个写入对象以写入修改后的文件
    postfix = info["format"]["format_name"].split(",")[0]
    output_file = ".".join(video_file.split(".")[:-1]) + ".processed." + postfix

    generate_output_video(video_file, output_file, input_fps, input_framesize, output_params)
