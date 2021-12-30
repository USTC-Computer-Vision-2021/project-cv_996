[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-f059dc9a6f8d3a56e377f745f24479a46679e63a5d9fe6f495e02850cd0d8118.svg)](https://classroom.github.com/online_ide?assignment_repo_id=6428153&assignment_repo_type=AssignmentRepo)

[toc]

# 基于Python-OpenCV的AR实现与优化

增强现实是将三维物体和相应信息映射在二维图像数据上的一系列操作的总称。通过从二维图像中获得三维信息、将三维模型映射到二维图像这两个基本操作，可以实现虚拟模型与现实照片的融合。



本实验报告分为两个部分，在第一部分中我们在Python-OpenCV平台实践了基于SIFT和orb算法进行特征匹配的增强现实算法。在第二部分中，我们总结了现有方案的部分缺点并提出了针对角点失配、模型抖动问题的解决方案。

------

成员及分工

- 周玉祺 PB18000280
  - 调研、编程、报告撰写
- 姜以恒 PB18000245
  - 调研、编程、报告撰写

## 增强现实

#### 1.原理概述

增强现实是将物体和相应信息放置在图像数据上的一系列操作的总称。简单的增强现实任务的目标可以概括为将一幅图像中标记物上的点映射到另一位置视角的图像的对应点上，解算出映射关系，再使用该解算出的映射矩阵将已有模型投影到原始图像中。

#### 2.算法流程

1.首先提取标记图像`model`和目标图像`frame`的SIFT或orb特征，再通过CV内置的匹配算法估计其单应性矩阵`homography`。

```python
if pair == "sift":
    kp_model, des_model = sift.detectAndCompute(model, None)
if pair == "orb":
    kp_model, des_model = orb.detectAndCompute(model, None)
...
if pair == "sift":
    kp_frame, des_frame = sift.detectAndCompute(frame, None)
if pair == "orb":
    kp_frame, des_frame = orb.detectAndCompute(frame, None)
...
if not (kp_frame == []):
    matches = bf.match(des_model, des_frame)
...
if len(matches) > MIN_MATCHES:         
    src_pts = np.float32([kp_model[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
```

------

2.结合照相机标定矩阵和单应性矩阵，可以获得两个视图之间的相对变换[2]。假设照相机标定矩阵为$K$,则在标记图像中，由于标记物在二维平面上($Z=0$)，其空间矩阵应为$P_1=K\left(\begin{array}{llll}1&0&0&0\\0&1&0&0\\0&0&1&-1\end{array}\right)$,经过单应性矩阵的变换，可以构造出目标图像的空间矩阵$P_2=HP_1$.

```python
def projection_matrix(camera_parameters, homography):
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    # normalise vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return np.dot(camera_parameters, projection)
```

------

3.依据解算出的目标图像的空间矩阵$P_2$，将3D模型投影目标图像中[1]。

```python
def render(img, obj, projection, model, color=False):
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 3
    h, w = model.shape

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        if color is False:
            cv2.fillConvexPoly(img, imgpts, (137, 27, 211))
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1]  # reverse
            cv2.fillConvexPoly(img, imgpts, color)

    return img
```

报告中展示的仅为最基本的算法框架，细节均在`/Python/src`中注释体现。

#### 3.实验结果

该方案下运行的实时AR实验结果在`/media/1.mp4`中展示。

## 优化方案

从`/media/1.mp4`中，我们可以总结出两个较为明显的问题：一是相机在特定角度会间歇性地解算出错误的投射矩阵使得模型脱离标记，而是模型就算能保持在标记上方，也会以一个非常高的频率抖动。

#### 1.角点失配

通过查看前十大匹配点我们发现，由于orb角点检测仅针对灰度图像，在一些特定的角度算法会错误地将一些环境中的“角”判断成目标。如下图所示：

![](/media/4.jpg)

图1：一个角点失配的实例

为了解决该问题，我们采取了一种简单的方案：通过设定颜色阈值，排除环境因素的影响。（该方案仅仅适用于标记图像颜色比较单一且与一般环境有较大差距的情况，如蓝色和绿色）。

我们首先依据颜色阈值获得一张二值化蒙版图像`mask`,再依据`mask`把阈值以外的区域设为白色（按照CV惯例的方法应该设为黑色，但是黑色会使标记点边缘的梯度方向全部反向，影响匹配性能，故应该设置成标记图像的背景色白色）。

```python
 if Colordetection:
    blur = cv2.GaussianBlur(frame, (7, 7), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    # standard: (35, 43, 46) -> (77, 255, 255)
    low_green = (35, 43, 46)
    high_green = (90, 255, 255)
    clmask = cv2.inRange(hsv, low_green, high_green)
    frame = cv2.add(np.zeros(np.shape(frame), dtype=np.uint8), frame, mask = clmask)
    frame = 255 - frame
```

采用颜色阈值之后的实验结果如视频`/media/2.mp4`所示。

![](/media/5.jpg)

图2：颜色阈值方法有效滤除了环境中的噪声干扰

#### 2.模型抖动

模型的抖动一般是由于摄像头采样率低和标记图像内有角点错配造成的，从图1中可以看出orb算法有时难以分辨箭头的左肩和右肩，因此模型虽然能吸附在标记上，但会出现左右、前后高频率摆动的情况。

为此，我们采用均值平滑方法来降低高频抖动。通过设置一个长度为10的队列窗口，将按时间序列输入的单应性矩阵与前9个取平均值后作为新的单应性矩阵。该方法可以有效缓解高频振动的问题，但对于快速运动的标记灵敏度会降低，出现延迟现象。

实验结果如视频`\media\3.mp4`所示。

## 图形界面

//todo

## 实验总结

在本实验中，我们基于Python-OpenCV平台，采用SIFT和orb方法实现了标记图像到摄像机视角的转换，并套用[1]的方法将三维模型映射到摄像机图像上，实现了动态AR算法。此后针对环境干扰和模型高频抖动问题，我们采用颜色阈值方法和均值滤波有效解决了问题，但仍有存在颜色阈值方法对较为复杂的标记图像不适用，均值滤波对标记物的高频运动灵敏度差等问题。

## 参考文献

[1]https://blog.csdn.net/weixin_41655918/article/details/89038462

[2]https://blog.csdn.net/limmmy/article/details/88973467

## 工程结构

```
│  README.md
│
├─.github
│      .keep
│
├─media
│      1.mp4
│      2.mp4
│      3.mp4
│      4.jpg
│      5.jpg
│
└─Python
    └─src
        │  ar_main.py
        │  objloader_simple.py
        │  objloader_simple.pyc
        │
        ├─.idea
        │  │  .gitignore
        │  │  misc.xml
        │  │  modules.xml
        │  │  src.iml
        │  │  vcs.xml
        │  │  workspace.xml
        │  │
        │  └─inspectionProfiles
        │          profiles_settings.xml
        │
        ├─models
        │      cow.obj
        │      ducky.obj
        │      fox.obj
        │      pirate-ship-fat.obj
        │      rat.obj
        │      wolf.obj
        │
        ├─reference
        │      model.jpg
        │      model2.png
        │      model3.png
        │      model4.png
        │      model5.jpg
        │
        └─__pycache__
                objloader_simple.cpython-36.pyc
                objloader_simple.cpython-37.pyc
```

运行环境：//todo



