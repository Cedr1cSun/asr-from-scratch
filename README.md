# ASR From-Scratch

四个小 ASR 模型全参数随机初始化(`Config(...) + Model(config)`,不加载任何预训练权重),在 LibriSpeech 上用 HuggingFace transformers 从零训练。每个模型交付 model / dataset / trainer 三件。

当前进度:**Whisper(medium)与 Parakeet-CTC 已完成并通过本地冒烟**;SenseVoice、X-ASR 未开始。

## 目录结构

```
asr-from-scratch/
├── common/                  # 四模型共用
│   ├── data.py              # LibriSpeech 流式抓取 + 本地缓存
│   └── metrics.py           # WER,文本归一化与 SURE-EVAL 的 wenet_compute_cer.py 逐位一致
├── whisper/                 # Whisper enc-dec + CE(Seq2SeqTrainer)
│   ├── model.py             # small/medium 尺寸预设,默认 medium(~764M)
│   ├── dataset.py           # collator:特征 pad 3000 帧,labels pad -100
│   ├── train.py             # python -m whisper.train [--lr --max-steps --run-name]
│   ├── smoke.py             # 冒烟轮 1:单条 overfit
│   ├── batch_probe.py       # 冒烟轮 3:batch 上限/步时探测
│   ├── reload_check.py      # save_pretrained 产物回载推理验证
│   └── config.yaml
├── parakeet/                # FastConformer + CTC(Trainer)
│   ├── model.py             # d=256 × 16 层,~26M
│   ├── dataset.py           # collator:变长 pad + attention_mask;labels pad 到 blank(1024)
│   ├── train.py / smoke.py / reload_check.py / config.yaml
└── tests/                   # pytest,含与平台 WER 脚本的交叉验证
```

## 环境

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

实测环境:python 3.12,torch 2.12.1+cu130,transformers 5.12.1,datasets 5.0,RTX 2080 Ti 22GB(WSL2)。

## 快速开始

一切命令从项目根目录跑:

```bash
# 冒烟轮 1:单条样本 FP32 overfit,loss→0 且贪心解码复现转写即 PASS(首跑自动下载 8 条样本缓存)
.venv/bin/python -m whisper.smoke          # ~1 分钟(--size small 更快)
.venv/bin/python -m parakeet.smoke         # ~10 秒

# 冒烟轮 2:mini100(100 条训练 + 20 条 eval,300 步)
.venv/bin/python -m whisper.train          # ~28 分钟
.venv/bin/python -m parakeet.train         # ~1 分钟

# 产物回载验证(镜像 SURE-EVAL ModelWrapper 的用法)
.venv/bin/python -m whisper.reload_check outputs/mini100_medium_fp32/final
.venv/bin/python -m parakeet.reload_check outputs/parakeet_mini100_fp32/final

# 测试
.venv/bin/python -m pytest tests/ -q
```

## 合规口径

- 模型权重全部随机初始化,任何路径都不 `from_pretrained` 权重
- tokenizer / feature_extractor / generation_config 复用官方 repo 的**配置文件**(查表与预处理参数,非神经权重):Whisper 用 `openai/whisper-{size}.en`,Parakeet 用 `nvidia/parakeet-ctc-0.6b`
- checkpoint 用标准 `save_pretrained` 目录(模型 + processor 同目录),可被 SURE-EVAL ModelWrapper 直接加载

## 冒烟结果摘要

| 项 | Whisper medium(764M) | Parakeet-CTC(26M) |
|---|---|---|
| 单条 overfit | 48 步 loss 11.1→0.01,解码复现 | 242 步 18.6→0.01,解码复现 |
| mini100(300 步) | lr 1e-4 单调收敛 7.77(< 随机基线 10.86);lr 3e-4 会震荡 | lr 3e-4 单调 14.6→5.64 |
| batch(FP32,2080 Ti) | bs2 无 ckpt 最优(1.07s/步);ckpt 开则 14.2G 平台到 bs16 | bs128 无压力,本地探不到上限 |
| 回载 | OK | OK |

## 注意事项

- datasets 5.x 音频解码默认走 torchcodec(需系统 FFmpeg);本项目用 `Audio(decode=False)` + soundfile 直接解 FLAC 绕开。
- from-scratch 的训练行为与微调完全不同:warmup 期 loss 冲高(Whisper 冲到 ~40)是常态,收敛判据看趋势;微调用的 token_acc 阈值不适用。
- 全量 960h 训练在集群做,建议 BF16;本地 FP32 只用于冒烟。

## 与 SURE-EVAL 的对齐

- WER 归一化逐行为移植平台 `evaluation/asr/wenet_compute_cer.py` 的英文路径(空白切词 → upper → 剥 `<tag>`,不删标点),`tests/test_metrics.py` 里与原脚本对同一批 ref/hyp 断言相等
- LibriSpeech test-clean/test-other 是平台原生数据集,训好的 checkpoint 走平台打分即可对比
