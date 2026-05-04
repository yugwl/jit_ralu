然后运行 RALU 版本生成 1 张图：

cd /home/cvip/deyu/jit_ralu/JiT

conda activate jit

CUDA_VISIBLE_DEVICES=0 python generate_one.py

生成结果会在：

/home/cvip/deyu/jit_ralu/JiT/result/sample_label0_ralu.png

如果想验证原版 JiT，不使用 RALU：

CUDA_VISIBLE_DEVICES=0 python generate_one.py --no_ralu
输出：

/home/cvip/deyu/jit_ralu/JiT/result/sample_label0_base.png