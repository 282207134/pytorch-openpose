import torch
from demo_video import input_fps
if torch.cuda.is_available():
    print(f"Torch device: {torch.cuda.get_device_name()}")
else:
    print("CUDA is not available. Check your PyTorch installation.")
print("是否可用：", torch.cuda.is_available())        # 查看GPU是否可用
print("GPU数量：", torch.cuda.device_count())        # 查看GPU数量
print("torch方法查看CUDA版本：", torch.version.cuda)  # torch方法查看CUDA版本
print("GPU索引号：", torch.cuda.current_device())    # 查看GPU索引号
print("GPU名称：", torch.cuda.get_device_name(1))    # 根据索引号得到GPU名称

