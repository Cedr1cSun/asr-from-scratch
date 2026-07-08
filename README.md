# ASR From-Scratch

四个小 ASR 模型全参数随机初始化(`Config(...) + Model(config)`,不加载任何预训练权重),在 LibriSpeech 960h 上用 HuggingFace transformers 从零训练。每个模型交付 model / dataset / trainer 三件,checkpoint 为标准 `save_pretrained` 目录,兼容 SURE-EVAL ModelWrapper。

当前状态(2026-07-08):**四个模型适配全部完成**(契约 v2,本地测试全绿);whisper 与 parakeet 的 960h full training 在集群进行中,sensevoice 与 x_asr 训练脚本已生成、等 GPU 配额开跑。

## 四个模型

| 包 | 结构 | 损失 | 规模 | tokenizer |
|---|---|---|---|---|
| `asrfs.whisper` | Transformer enc-dec(whisper medium 尺寸) | CE(Seq2SeqTrainer) | ~764M | 官方 whisper tokenizer(查表资产,钉 revision) |
| `asrfs.parakeet` | FastConformer,d=256 × 16 层 | CTC | ~26M | ParakeetTokenizerFast(HF 资产,钉 revision) |
| `asrfs.sensevoice` | SANM(FSMN 注意力),d=384 × 16 层 | CTC | ~50M | 同 parakeet(钉 revision) |
| `asrfs.x_asr` | Zipformer-S(icefall 组件 vendor 进仓) | RNN-T(torchaudio rnnt_loss) | ~23M | 自训 BPE500(`asrfs/x_asr/bpe/librispeech_bpe500.model`,入 git) |

统一契约(每包 `__init__.py` 导出):`build_model / build_processor / build_collator / build_dataset / make_example / decode / load_checkpoint / save_checkpoint / build_trainer`,外加 `LOSS_FAMILY`(ce/ctc/rnnt,harness 六断言与 smoke 判据按此分家)。

## 目录结构

```
asr-from-scratch/
├── asrfs/
│   ├── common/            # 四模型共用
│   │   ├── full_data.py   # 960h 数据线:HF / 公司 manifest 双源,预计算特征 + speed perturb ×3
│   │   ├── augment.py     # SpecAugment(dataset transform,训练时启用)
│   │   ├── ctc.py         # CTC 族共用 collator/decode/trainer
│   │   ├── data.py        # 冒烟用小数据线(流式抓取 + 本地缓存)
│   │   └── metrics.py     # WER,归一化与 SURE-EVAL wenet_compute_cer.py 逐位一致
│   ├── whisper/ parakeet/ sensevoice/ x_asr/
│   │   ├── model.py dataset.py train.py
│   │   ├── config.yaml        # 冒烟(mini100)超参
│   │   └── config_full.yaml   # 960h 全量超参
│   └── x_asr/_vendor/     # icefall zipformer encoder/decoder/joiner(来源见 VENDOR.md)
├── scripts/eval_full.py   # 离线评测:任意 checkpoint × dev-clean/test-clean/test-other → WER
├── tests/                 # pytest;含契约测试与平台 WER 脚本交叉验证
└── docs/                  # 设计文档、运行记录
```

## 环境

### 本地(开发/冒烟)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .          # asrfs editable 安装,供 asr-harness 同 venv import
```

### 集群(960h 训练,实测配方)

驱动 CUDA 12.8,BF16 卡(A10/3090/4090);外网可通,HF 走 hf-mirror:

```bash
python3 -m venv ~/envs/asrfs
~/envs/asrfs/bin/pip install torch==2.12.1+cu126 torchaudio==2.11.0+cu126 \
    -f https://mirrors.aliyun.com/pytorch-wheels/cu126/
~/envs/asrfs/bin/pip install "datasets[audio]==5.0.0" transformers==5.12.1 \
    evaluate==0.4.6 jiwer==4.0.0 accelerate==1.14.0 soundfile==0.14.0 \
    pyyaml==6.0.3 tensorboard==2.21.0 librosa==0.11.0 sentencepiece==0.2.1 \
    numpy==2.4.6 huggingface_hub==1.21.0 jsonschema==4.26.0 pytest==9.1.1 packaging==26.2
~/envs/asrfs/bin/pip install -e ~/WorkSpace/asr-from-scratch

export HF_ENDPOINT=https://hf-mirror.com     # 所有 HF 访问(tokenizer 资产、eval split)都要
```

5090(sm_120)需换 cu128 轮子(同镜像站换 `cu128/` 路径),未实测训练。

## 数据准备(预计算特征)

训练数据两条线,由 `config_full.yaml` 的 `data.source` 选择:

- `hf`:HF `openslr/librispeech_asr`(钉 revision),三个 train split
- `manifest`:公司 jsonl 清单(每行 `{path, target}`,path 指向集群 wav),拼成单一 `train.960` split;manifest 内容 md5 入数据指纹

eval split 恒为 HF `validation.clean`,不随源分叉。训练 split 做 speed perturb ×3(0.9/1.0/1.1,eval 不做),SpecAugment 在训练时以 dataset transform 施加。

```bash
export ASRFS_DATA_DIR=/path/to/asrfs-full     # 特征落盘根目录
python -m asrfs.common.full_data --config asrfs/<model>/config_full.yaml --adapter asrfs.<model>
```

960h 单模型特征约数十 GB;超参/tokenizer 变更会改变 `params_hash`,旧特征自动判 stale 重算。

## 冒烟与测试(本地)

```bash
# whisper/parakeet 自带单模型冒烟入口
.venv/bin/python -m asrfs.whisper.smoke
.venv/bin/python -m asrfs.parakeet.smoke

# 四模型完整冒烟(overfit1/mini100/batch 探测/回载,11-stage 管线)走 asr-harness:
#   见 asr-harness/manual.md(executors/run_smoke.py 等)

.venv/bin/python -m pytest tests/ -q
```

## 960h Full Training(集群)

管线经 asr-harness 生成与执行:

```bash
cd ~/WorkSpace/asr-harness
python executors/gen_training_script.py \
    --adapter-pkg asrfs.<model> \
    --config ~/WorkSpace/asr-from-scratch/asrfs/<model>/config_full.yaml \
    --run-id fullrun-<model> \
    --manifest $ASRFS_DATA_DIR/full/<model>/manifest.json \
    --n-gpu 4 --data-root $ASRFS_DATA_DIR
# 产物:outputs/fullrun-<model>/train_full.sh(torchrun 4 卡,bf16)
bash outputs/fullrun-<model>/train_full.sh
```

volcano 集群提交示例(4×GPU,32C/128G):

```bash
vc submit -t pytorch -p <partition> -j rc-train-<model> -n 1 -c 32 -m 128G -g 4 \
    -i <docker-image> -v /hpc_stor03:/hpc_stor03 -v /hpc_stor08:/hpc_stor08 \
    -cluster D6 --cmd 'bash ~/train_wrapper.sh <model>'
```

### 训练监控要点

- 日志里的 loss 是梯度累积求和口径:除以 `gradient_accumulation_steps` 才是 batch 均值。
- **CTC 族(parakeet/sensevoice)必须用解码探针盯 blank fraction,不能只看 loss**:loss 正常下降也可能已经 blank collapse(全输出 blank,loss 停在 blank 地板附近)。教训:parakeet 首跑 LR 3e-4 在 13k 步塌缩;药方 LR 1.5e-4 + warmup 5000 + 有效 batch 256 复训后,checkpoint-1000 处 blank ≈ 0.75 为健康轨迹。
- 探针判据:blank < 0.95 为脱离塌缩;step ≥ 10000 仍 ≥ 0.98 判死重开。

## 评测

任意 checkpoint × 任意 eval split,特征在线抽取(不依赖预计算),WER 用 wenet 归一化口径(与 SURE-EVAL 一致):

```bash
cd asr-from-scratch
python scripts/eval_full.py <model> <ckpt_dir> {validation.clean|test.clean|test.other} \
    [--limit N] [--batch B] [--device cuda:0]
```

注意:HF Trainer 的中途 `checkpoint-N` 只含裸模型权重(tokenizer/spm 资产在训练结束的最终 save 才写入;whisper 例外)。脚本对此自动回退:结构从 config 重建 + safetensors 载权重 + processor 从仓内钉死资产重建,与训练严格同源。
