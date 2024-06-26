# pose estimation with mediapipe
mediapipe是一个开源的机器学习框架，可以用来做pose estimation，即识别人体的关节和手部的位置。

本项目基于mediapipe，用python实现了简单的pose estimation功能。

## 文件描述
* advanced_camera.py: 高阶（？）摄像头实时检测（实时处理，并给关键点添加颜色、改变大小）
* simple_camera.py: 普通版摄像头实时检测（实时处理）

## 解锁功能
* 摄像头实时读取每一帧
* 关键点的样式更改（大小、颜色）

## 运行
1. 建议先读懂代码，注释基本上都很详细。
2. 配置环境，*建议用conda开一个虚拟环境，在这个虚拟环境里配相应的环境（非必要，不想了解可以不用管这个）*
   1. 保证有python环境
   2. 导入mediapipe：在终端输入`pip install mediapipe`
   3. 看代码里import了什么，就用pip下载什么就好了，最好先搜一下导入的库需要用什么命令来导入。例如，`import cv2`对应的安装命令是`pip install opencv-python`
3. 运行代码
   1. 可以直接在终端输入`python {需要运行的那个代码}`
   2. 涉及到摄像头的部分需要注意不同系统的区别，具体见代码注释
   3. 按下键盘的 q 或 esc 退出（在英文输入法下）