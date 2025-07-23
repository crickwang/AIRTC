# TTS 概述

## edge-TTS

[github](http://github.com/rany2/edge-tts)

---

edge-TTS 是一个基于 Microsoft Edge 的TTS引擎。它并不是一个AI模型。

### edge-TTS 音质
与edge浏览器的语音播报相同，音质一般，支持多种语言和语音，但缺乏有情感的音调。声音较为单一，适合用于简单的语音提示和朗读场景。

### edge-TTS 延迟
edge-TTS 的延迟理论上较低，具体取决于网络状况和文本长度。

### edge-TTS 算力
原文并未提及，但由于它是基于浏览器的实现，应该算力需求较低。

## piper

[github](https://github.com/rhasspy/piper)

---

piper 是一个基于 Tacotron 2 和 WaveRNN 的 TTS 模型，旨在提供高质量的语音合成。它支持多种语言和声音，并且可以通过训练自定义声音。

### piper 音质
piper 的音质较高，能够生成流畅的语音，但无法正确地断句,音调相当单一。

### piper 延迟
piper 的延迟相对较低。

### piper 算力
作为一个本地的小型模型，它对于算力的需求较低，可以在普通的 CPU 上运行。

### piper 协议
piper 使用 MIT 许可证。

## ChatTTS

[github](https://github.com/2noise/ChatTTS)

---

### chatTTS 音质
chatTTS 音质极高，支持多种语言和声音，能够生成自然流畅的语音。它还支持情感语音合成，可以根据文本内容调整语调和情感。

### chatTTS 延迟
chatTTS 的延迟相当高，根据[issue](https://github.com/2noise/ChatTTS/issues/279)，chatTTS 无法做到流式输出，甚至伪流式也相当困难。

### chatTTS 算力
chatTTS 对算力的需求较高，通常需要在 GPU 上运行以获得较好的性能。

## kokoro

[github](https://github.com/hexgrad/kokoro)

---

Kokoro is an open-weight TTS model with 82 million parameters.

### kokoro 音质
Kokoro's音质较高，支持中文和英文，但无法混合多种语言同时生成。在单一语言的情况下，其能够生成自然流畅的语音，但情感音调方面欠佳。

### kokoro 延迟
Kokoro 的延迟相对较低，支持流式输出。

### kokoro 算力
Kokoro 对算力的需求一般，其82m的参数并不算大。

### kokoro 协议
Kokoro 使用 Apache 2.0 许可证。

## 其余模型

更多关于开源文本到语音模型的信息，请参阅以下[链接](https://waytoagi.feishu.cn/wiki/E5aGwN9PSie8i3kwBvgcAXoPnDf),其中所有的模型均为音质极佳，情感流畅的开源模型，但毫无例外地有较高地延迟，无法进行实时地流式输出。

在这篇[文章](https://www.bentoml.com/blog/exploring-the-world-of-open-source-text-to-speech-models)中，作者对开源TTS模型进行了详细的对比和分析，涵盖了音质、延迟、算力需求等方面，但并未提及与延迟有关地内容。

## 总结

综合来看，kokoro 是一个性能均衡的 TTS 模型，适合对音质和延迟有一定要求的应用场景，其更适合这个项目的需求。
