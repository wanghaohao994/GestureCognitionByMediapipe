# 基于MLX的手势控制系统

## 项目简介
这是一个基于神经网络的手势识别与控制系统，可以通过摄像头捕捉手部动作并将其转换为电脑控制命令。

## 功能特点
- 实时手势检测与识别
- 支持鼠标移动、点击控制
- 支持缩放功能（手势控制Ctrl+/Ctrl-）
- 支持页面滚动功能

## 环境要求
- Python 3.11
- OpenCV
- MediaPipe
- PyTorch
- PyAutoGUI
- Pygame

## 安装说明
```bash
# 创建并激活环境
conda create -n gcs python=3.11
conda activate gcs

# 安装依赖
pip install -r requirements.txt
```

## 使用方法
```bash
python run_NN_CPU_env.py
```

## 支持的手势
- 手势0：点击
- 手势1：鼠标移动
- 手势2：缩放控制
- 手势3：页面滚动