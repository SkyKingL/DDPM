import os

# 列出 GPU 设备信息
# nvidia-smi -L
# >> GPU 0: NVIDIA GeForce GTX 1080 Ti (UUID: GPU-ed5fc99e-eb41-fa51-7392-cd219643ab83)
# >> GPU 1: NVIDIA GeForce GTX 1080 Ti (UUID: GPU-4042dbf9-0217-99f9-f3cf-b52188f8bad5)
# >> GPU 2: NVIDIA GeForce GTX 1080 Ti (UUID: GPU-9f6c6ccc-d575-8268-dc18-7523906897a6)
# >> GPU 3: NVIDIA GeForce GTX 1080 Ti (UUID: GPU-3a29a1a1-28ac-7073-8412-aac2397cea3d)

SEED = 514
DETERMINISTIC = False # 随机数种子seed确定时，模型的训练结果将始终保持一致。但速度可能会慢一点。。平时训练就用False好了

IMAGE_SIZE = 64
BATCH_SIZE = 8 # 如果out of memory了，就设小一点。。
TIMESTEPS = 1000
DATA_DIR = "church_outdoor_train"
DEVICES = [0,1,2,3] # DEVICES可以是"auto" 表示自动选择gpu,可以是 1 2 3这种数，表示用几个gpu,也可以是一个列表 里面写[0,1,2]表示用哪几个gpu
MAX_EPOCHES = 500

DATALOADER_WORKERS = os.cpu_count() # DATALOADER_WORKERS是开多少个进程来加载数据,这个是cpu用的进程,推荐是设成cpu核心数
# DATALOADER_WORKERS = 2 

# SAVE_EVERY_N_EPOCHS = 10
SAVE_EVERY_N_EPOCHS = None 

SAVE_EVERY_N_STEPS = 1000 
# SAVE_EVERY_N_STEPS = None
