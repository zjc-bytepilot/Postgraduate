🌟 OpenPan: A Data-Adaptive and Band-Agnostic Unsupervised Pansharpening Framework
OpenPan 是一个高度模块化、配置驱动的遥感全色锐化（Pansharpening）深度学习框架。本框架致力于打破传统深度学习在遥感融合任务中面临的“波段壁垒”与“域偏置”痛点，通过结合传统地统计学的物理先验与深度学习的强拟合能力，实现极致的跨传感器泛化性能。

✨ 核心特性 (Key Features)
🧩 高度解耦的工程架构 (OpenMMLab-Style)

采用全局注册器（Registry）机制，将 Backbone, Neck, Head, Loss 与 Dataset 彻底解耦。

无需修改代码，仅通过 YAML 配置文件即可自由拼装网络、切换数据集与超参数，实现“乐高式”炼丹。

🌌 波段不可知设计 (Band-Agnostic Architecture)

输入端：采用波段展平与共享权重映射，网络不再受限于固定的波段数。

输出端：利用光谱能量分布特征进行单通道高频细节的动态广播（Dynamic Broadcasting）。

优势：同一套权重可无缝处理 GF-2（4波段）、Landsat（7波段）或 WV-3（8波段）数据，实现 Zero-Shot 跨卫星推理。

🧠 统计特征驱动的动态卷积 (Statistic-Driven Dynamic Convolution)

引入场景特征提取器，实时计算输入数据的全局光谱与空间先验（均值、方差、梯度能量）。

基于先验特征动态生成（CondConv）当前图像专属的专家卷积核权重，彻底摆脱传统 CNN 静态权重对特定数据集的过拟合，赋予网络类似传统地统计算法（如 ATPRK）的强自适应泛化能力。

📐 基于物理约束的无监督学习 (Physics-Informed Unsupervised Loss)

摒弃暴力的 AvgPool，采用基于卫星传感器 MTF（调制传递函数）的高斯低通滤波进行物理退化降采样。

结合 Wald 准则，无需全分辨率参考图像（Ground Truth）即可进行端到端的无监督训练。

📊 全栈式权威指标评估体系 (Comprehensive Metrics)

内置遥感领域的黄金评价标准：PSNR, SSIM, SAM (光谱角映射), ERGAS, CC (相关系数), RMSE, UIQI。

🛠️ 安装与环境依赖
推荐使用 Anaconda 构建虚拟环境：

Bash
conda create -n pansharpening python=3.9
conda activate pansharpening
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install h5py pyyaml tensorboard tqdm scikit-image matplotlib
📂 目录结构
Plaintext
OpenPan/
├── configs/                              # ⚙️ YAML 配置文件目录
│   └── unsupervised_dynamic_pan.yaml 
├── openpan/                              # 🧠 核心算法库
│   ├── registry.py                       # 注册器枢纽
│   ├── datasets/                         # 数据加载器 (H5Dataset)
│   ├── evaluation/                       # 评价指标 (metrics.py)
│   ├── models/                           
│   │   ├── backbones/dynamic_conv.py     # 统计特征提取与动态卷积主干
│   │   ├── necks/sft_neck.py             # 光谱引导注入颈部 (SFT)
│   │   ├── heads/dynamic_head.py         # 波段动态能量重组头
│   │   ├── losses/unsupervised_loss.py   # 基于 MTF 的物理无监督损失
│   │   └── framework.py                  # 网络组装外壳
│   └── engine/trainer.py                 # 工业级训练引擎 (断电恢复/日志/TensorBoard)
├── tools/                                # 🚀 执行脚本
│   ├── train.py                          # 统一训练脚本
│   ├── test.py                           # 定量指标测试脚本
│   └── test_visualize.py                 # 假彩色可视化脚本
└── README.md
💾 数据准备
本框架使用 .h5 格式作为数据输入，要求内部包含 ms, pan 以及可选的 gt 键值。

⚠️ Windows 环境极度重要警告 (HDF5 C-Library Bug):
由于 HDF5 底层 C 语言库的历史遗留问题，H5 文件的绝对路径中绝对不能包含任何中文字符！ 否则会报 No such file or directory 错误。

请确保您的数据存放于纯英文路径下，并在 configs/*.yaml 中配置正确的绝对路径，例如：

YAML
dataset_train:
  type: PansharpeningH5Dataset
  data_path: "D:/Nanqing/DeepLearning/Landsat_Pansharpen/datasets_h5/pansharpening_gf2/training_data/train_gf2.h5"
  max_value: 1023.0  # GF-2 通常为 10-bit 数据 (极值1023)
  normalize_mode: '0_1'

dataset_val:
  type: PansharpeningH5Dataset
  data_path: "D:/Nanqing/DeepLearning/Landsat_Pansharpen/datasets_h5/pansharpening_gf2/validation_data/valid_gf2.h5"
  
dataset_test:
  type: PansharpeningH5Dataset
  data_path: "D:/Nanqing/DeepLearning/Landsat_Pansharpen/datasets_h5/pansharpening_gf2/testing_data/test_gf2.h5"
🚀 快速开始 (Getting Started)
1. 训练模型 (Training)
引擎支持自动断电恢复、定期验证与 TensorBoard 记录。

Bash
# 确保在 OpenPan 根目录下执行，并将当前路径加入环境变量
set PYTHONPATH=%PYTHONPATH%;%cd%  # Windows写法 (Linux: export PYTHONPATH=$PYTHONPATH:$(pwd))

python tools/train.py --config configs/unsupervised_dynamic_pan.yaml
实时监控训练状态： tensorboard --logdir ./work_dirs/unsupervised_dynamic_pan/tf_logs

2. 定量测试 (Evaluation)
遍历测试集并计算 7 大权威遥感指标（PSNR, SSIM, SAM, ERGAS, CC, RMSE, UIQI）。

Bash
python tools/test.py
3. 定性可视化 (Visualization)
加载最优模型，输出高分辨率 PAN、原始 MS 以及预测的 HRMS，并执行专业的百分比截断线性拉伸，生成 NIR-Red-Green 假彩色合成图 供论文对比使用。

Bash
python tools/test_visualize.py
📝 待办事项 (TODO)
[x] 支持跨波段 Zero-shot 推理

[x] 引入基于 MTF 的物理退化过程

[x] 实现独立于主代码的评估指标库解耦

[ ] 补充对 TIFF 格式遥感影像的直接推理支持
