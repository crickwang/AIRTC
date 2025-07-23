# ASR 概述

## Whisper cpp based on OpenAI Whisper

[openai github](https://github.com/openai/whisper) | [openai paper](https://arxiv.org/pdf/2212.04356) | [openai demo](https://huggingface.co/spaces/openai/whisper) | [whisper.cpp github](https://github.com/ggml-org/whisper.cpp)

---

Whisper is a general-purpose speech recognition model trained on 680,000 hours of multilingual and multitask supervised data collected from the web. It is designed to perform well across a variety of tasks, including transcription, translation, and language identification.

需要注意的是，whisper的设计目标是通用性和多任务处理，而不是专门针对低延迟或实时应用进行优化。因此，虽然它在准确率上表现出色，但在延迟方面可能不如一些实时应用设计的模型。并且whisper并不是流式ASR（streaming ASR），并不符合项目要求。但是，在此模型基础上进行流式处理（whisper.cpp）可以做到伪流式（peudo-streaming）的，并且此开源模型理论上与whisper有相同的准确度。

### Whisper & Whisper cpp 准确率对比表

> Word Error Rate (WER) is a metric of the performance of an automatic speech recognition (ASR) system. This value indicates the percentage of words that were incorrectly predicted. 

#### 1. 英语语音识别（LibriSpeech + 跨域测试）

| 模型/主体                     | LibriSpeech test-clean WER↓ | 跨域数据集平均 WER↓ | 相对人类差距 |
| **人类转录（Alec）**          | ~5.8%（估计）               | ~10-15%             | 基准         |
| 传统微调模型（LibriSpeech专训） | **1.4%**（SOTA）           | ~30%+               | 跨域明显变差 |
| Whisper Large（Zero-shot）   | **2.5%**                   | **12.8%**（少55%错误） | 接近人类 |

---

#### 2. 跨域数据集（12个测试集平均结果）

| 模型                       | 参考集 WER↓ (LibriSpeech) | 其他域平均 WER↓ | 错误减少率 |
|----------------------------|---------------------------|-----------------|------------|
| wav2vec2.0 Large           | 2.7%                     | 29.3%           | 0%（基线） |
| Whisper Large (Zero-shot)  | 2.7%                     | **12.8%**       | **55.2% 减少** |

---

#### 3. 多语言识别（MLS、VoxPopuli、Fleurs）

| 数据集                      | Whisper Zero-shot WER↓ | 对比模型                    | 结果     |
|-----------------------------|------------------------|-----------------------------|----------|
| MLS（Multilingual LibriSpeech） | **7.3%**             | XLS-R(10.9%) / mSLAM(9.7%)  | **最好** |
| VoxPopuli                   | 13.6%                 | Maestro(8.1%) / mSLAM(9.1%) | 略逊     |
| Fleurs（96语言）            | WER与训练数据量强相关  | —                           | 高资源好，低资源差 |

---

#### 4. 语音翻译（CoVoST2 X→英语 BLEU↑）

| 任务                  | Whisper Zero-shot | mSLAM | Maestro |
|-----------------------|-------------------|-------|---------|
| 低资源语言翻译 BLEU↑ | **+6.7 BLEU 提升**| 中等  | 略弱    |
| 高资源语言翻译 BLEU↑ | 略低              | 略高  | 最优    |

---

#### 5. 噪声环境下表现（LibriSpeech test-clean 加噪）

- 信噪比 **<10dB**（酒吧噪音）下：  
  - **Whisper错误率最低**  
  - LibriSpeech微调模型在强噪音下性能崩溃  

---

#### 6. 人类 vs Whisper 对比（Kincaid46 数据集 25段音频）

| 转录方式                 | WER↓ |
|--------------------------|------|
| 纯人工转录               | ~8-10% |
| 计算机辅助+人工校对       | **最低 ~7-8%** |
| Whisper Large Zero-shot  | **~8-9%（仅差0.5-1%）** |

### Whisper cpp 延迟

并未提及到延迟，但由于whisper.cpp是伪流式处理，延迟会比真正的流式ASR高。根据[issue](https://github.com/ggml-org/whisper.cpp/issues/10)讨论和[cpp repo](https://github.com/ggml-org/whisper.cpp)的实现，whisper.cpp每0.5秒读取一定时间（whisper的初始参数为30秒）的音频片段进行处理并转录，最终结果*似乎*在最初停滞一段时间之后可以做到流畅的输出。

### Whisper cpp 部署
部署相对简单，支持多个平台（Apple Neural Engine (ANE) via Core ML, OpenVINO, NVIDIA GPU, and so on）。可以离线运行。

### Whisper cpp 社区支持
whisper 为 openai 提供的开源模型，拥有大量的社区支持和文档。

### Whisper cpp 大小
whisper.cpp是一个无法训练和微调的模型，主要用于推理。其大小基本取决于weight size。根据使用的模型的大小，large whisper模型约需要 **4GB**内存, 在量化后模型大小可以进一步缩小。

| 模型   | 磁盘大小 (Disk) | 内存占用 (Mem) |
|--------|----------------|----------------|
| tiny   | 75 MiB         | ~273 MB        |
| base   | 142 MiB        | ~388 MB        |
| small  | 466 MiB        | ~852 MB        |
| medium | 1.5 GiB       | ~2.1 GB        |
| large  | 2.9 GiB       | ~3.9 GB        |

### Whisper cpp 缺点
- **延迟较高**：由于是伪流式处理，延迟不如真正的流式ASR。
- **只能推理**：whisper.cpp目前仅支持推理，不支持训练和微调。
- **幻觉严重**：根据[社区反馈](https://community.openai.com/t/whisper-hallucination-how-to-recognize-and-solve/218307)和[新闻报道](https://www.healthcare-brew.com/stories/2024/11/18/openai-transcription-tool-whisper-hallucinations)，whisper在某些情况下会产生严重的幻觉（hallucination）。
- **语义不完整**：whisper cpp每0.5秒读取一定时间（如2秒）的音频片段进行处理， 导致每段音频的上下文信息可能不完整（被截断的单词或声调等），从而影响生产的语义完整性。目前并未在issue中看到是否已经解决此问题。

### Whisper cpp 协议
whisper.cpp使用MIT许可证。

==============

## FunASR with paraformer

[FunASR github](https://github.com/modelscope/FunASR) | [FunASR paper](https://arxiv.org/pdf/2305.11013) | [paraformer paper](https://arxiv.org/pdf/2206.08317)

---

FunASR是一个基础语音识别工具包，提供多种功能。FunASR提供了便捷的脚本和教程，支持预训练好的模型的推理与微调， 其中旗舰的模型为Paraformer。Paraformer是达摩院语音团队提出的一种高效的非自回归端到端语音识别框架。Paraformer is a non-autoregressive end-to-end speech
recognition model that has been trained on a manually annotated Mandarin speech recognition dataset that contains 60,000 hours of speech.

## FunASR 准确度

> Character Error Rate (CER) is a metric of the performance of an automatic speech recognition (ASR) system. This value indicates the percentage of characters that were incorrectly predicted. 

| 模型                | AISHELL1 CER↓ | AISHELL2 CER↓ | WenetSpeech CER↓ |
|---------------------|---------------|---------------|------------------|
| Conformer (ESPNET)  | 4.60%         | 5.70%         | 15.90%           |
| WeNet Conformer-U2++| 4.40%         | 5.35%         | 17.34%           |
| PaddleSpeech DS2    | 6.40%         | –             | –                |
| K2 Transducer       | 5.05%         | 5.56%         | 14.44%           |
| **Paraformer**      | 4.95%         | 5.73%         | –                |
| **Paraformer-large**| **1.95%**    | **2.85%**    | **6.97%**        |

## FunASR 延迟

> RTF (Real-Time Factor) 越小表示推理越快，<1 表示快于实时。AR = 自回归，NAR = 非自回归（paraformer）。

| 模型类型       | RTF↓    |
|----------------|---------|
| Conformer AR   | 0.21~1.43 |
| Paraformer     | **0.0168** |
| Paraformer-large | **0.0251** |

## FunASR 部署
FunASR支持多种部署方式，包括ONNX Runtime、TensorRT和Libtorch。它可以在CPU、GPU以及移动端（Android/iOS）上运行。FunASR还提供了模块化支持，可以根据需要选择不同的ASR模型和VAD模块。支持训练和微调。支持流式和非流式ASR。

## FunASR 社区支持
FunASR是阿里达摩院的开源项目，拥有活跃的社区支持和文档。

## FunASR 大小
N/A

## FunASR 缺点
- **社区支持较少**：相比于Whisper，FunASR的社区支持和文档相对较少。
- **大小未知**：FunASR的模型大小和内存占用没有明确的文档说明，可能需要根据具体模型进行测试。
- **未经广泛测试**：FunASR虽然提供了多种功能，但它作为一个开项不到一年，目前还在修bug以至于[issue](https://github.com/modelscope/FunASR/issues)中还有人在求助的项目，其实际应用中的表现和稳定性可能需要更多的测试和验证。

### FunASR 协议
FunASR使用MIT许可证。

============

## Vosk

[vosk github](https://github.com/alphacep/vosk-api) | [vosk website](https://alphacephei.com/vosk/)

---

vosk是一个基础语音识别工具包。网络上没有太多关于它的资料。

## Vosk 准确度
Accuracy of modern systems is still unstable, that means sometimes you can have a very good accuracy and sometimes it could be bad. It is hard to make a system that will work good in any condition. And there could be many reasons for that.

## Vosk 延迟
N/A

## Vosk 部署
目前看见python， java， c#， js，web， 和ios（需要联系开发者）的部署方式。
似乎只支持Python，其他语言的支持情况不明。

## Vosk 社区支持
Vosk是一个开源项目，社区支持和文档相对较少。它的GitHub页面有一些issue和讨论，但整体活跃度远不如Whisper或FunASR。

## Vosk 大小
最小的模型需要300MB的内存和50MB的磁盘空间。

## Vosk 缺点
- **准确度不稳定**：Vosk的准确度在不同条件下可能会有所波动，这使得它在某些应用场景下可能不够可靠。
- **社区支持较少**：相比于Whisper和FunASR，Vosk的社区支持和文档相对较少。
- **资料有限**：vosk的网络资料较少，并未提及延迟相关问题。

## Vosk 协议
Vosk使用Apache 2.0许可证。

============

## 总结
- **Whisper cpp**：适合需要高准确率和多语言支持的应用，但延迟较高，且不是真正的流式ASR。
- **FunASR**：提供了高效的非自回归ASR模型，但作为新的开源项目，仍存在相当大量的bug和社区支持不足的问题。
- **Vosk**：信息稀缺，准确度不稳定，社区支持较少，相较其余两种项目无优势。

因此，推荐使用 **Whisper cpp** 作为流式ASR的基础模型。这是目前[公认](https://www.sciencedirect.com/science/article/pii/S2666307424000573)的适合ASR的开源模型，具有较高的准确率和多语言支持。虽然延迟较高，但可以通过伪流式处理来满足项目需求。


