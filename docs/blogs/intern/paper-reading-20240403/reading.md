## NeRF, Gaussian Splatting and EG3D

> 2024 / 04 / 03

### 概念理解

#### NeRF

paper: [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](http://arxiv.org/abs/2003.08934)

website: https://www.matthewtancik.com/nerf

code: [NeRF-pytorch](https://github.com/yenchenlin/nerf-pytorch.git)

Neural Radiance Fields，神经辐射场，仅用 2D 的 posed images 作为监督，即可表示复杂的三维场景，用于新视角合成。

Radiance Field 指的 是一个函数，它将一个点和一个方向映射到光线颜色上，可以预测在三维空间中的任何一个位置。换言之，如果你站在某个位置，并且看向某个特定方向，那么 Radiance Field 就决定了你看到的景象。论文中构建了一个包含 3 维的位置信息以及 2 维视角信息的辐射场，并且使用了基本的 MLP 网络进行表征。

NeRF 工作的基本流程可以总结如下：

- 使用一系列2D图片作为输入数据集，这些图片显示了同一场景从不同角度和位置拍摄得到的结果
- 首先会估计出每张图片对应相机参数(如位置、焦距等)
- 然后利用这些参数以及深度信息将2D像素转换为3D空间坐标和视线方向,并通过神经网络对Radiance Field进行学习

<img src="./reading.assets/image-20240403103053121.png" alt="image-20240403103053121" style="zoom:33%;" />

训练时，NeRF 会尝试描述一个 3D 场景，并尝试渲染图像进行训练。

生成新的图片时，首先需要确定视线的方向，图像上的每一个像素对应一条 ray，ray 在穿过 Radiance Field 时，会经过多个采样点，每个采样点包含其颜色信息以及透明度信息 (是否被遮挡)。

为优化 MLP 网络，论文中还提出了两种优化方法： **Positional encoding** 和 **Hierarchical volume sampling**。

#### Gaussian Splatting