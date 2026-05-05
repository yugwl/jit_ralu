JiT/
├── README.md               # 项目的主说明文档，包含项目简介、安装和运行指南
├── LICENSE                 # 项目的开源许可证文件
├── environment.yaml        # Conda 环境配置文件，用于一键安装项目所需的依赖包
├── use.md                  # 用户使用指南或操作说明记录
├── analysis.md             # 实验分析或相关方法探讨的笔记文档
├── goal.md                 # 记录项目目标、里程碑或开发计划的文档
├── stage2_acceleration_solutions.md # 关于第二阶段加速方案的探讨和记录说明
│
├── main_jit.py             # 项目的主入口脚本，通常用于启动模型训练或全局评估
├── engine_jit.py           # 核心训练/验证引擎，包含单个 Epoch 的训练循环、损失计算、优化步进等逻辑
├── model_jit.py            # JiT (Just image Transformer) 核心模型架构的定义文件
├── denoiser.py             # 扩散模型中的去噪器 (Denoiser) 定义文件，负责执行去噪扩散过程
├── generate_one.py         # 用于使用训练好的模型推理并生成单张/少量图像的脚本
├── generate_one_diag.py    # 带有诊断信息输出的图像生成脚本，可能用于分析生成过程的中间状态
├── prepare_ref.py          # 数据准备脚本，通常用于准备计算 FID 等指标所需的参考真实数据集
│
├── demo/                   # 存放用于 README 展示和测试说明的示例图片或演示文件
│
├── fid_stats/              # 存放用于评估图像生成质量 (FID 指标) 的预计算统计文件
│   ├── jit_in256_stats.npz # ImageNet 256x256 分辨率的真实数据统计特征
│   └── jit_in512_stats.npz # ImageNet 512x512 分辨率的真实数据统计特征
│
├── result/                 # 默认的输出文件夹，用于保存模型权重 (checkpoints)、日志文件或生成的中间图像
│
└── util/                   # 工具类文件夹，包含各种辅助功能代码
    ├── crop.py             # 图像裁剪及相关预处理工具函数
    ├── lr_sched.py         # 学习率调度器 (Learning Rate Scheduler) 定义，如 Warmup 或 Cosine 衰减
    ├── misc.py             # 杂项工具函数，包含分布式训练设置、日志记录、度量平均等通用功能
    └── model_util.py       # 模型相关的辅助函数，例如模型权重加载、初始化或参数量统计等