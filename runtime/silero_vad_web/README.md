# Silero VAD + WeNet 实时语音识别系统

## 项目概述

本项目集成了 Silero VAD（Voice Activity Detection，语音活动检测）和 WeNet 端到端语音识别系统，实现了实时流式语音识别功能。

### 主要特性

- **实时语音检测**：使用 Silero VAD 自动检测语音活动
- **流式识别**：每识别出一个语音片段立即返回结果
- **Web 界面**：提供友好的网页界面，支持开始/停止录音
- **参数可调**：支持 VAD 所有关键参数的实时调节
- **可视化**：音频可视化和状态指示器

## 环境要求

### 服务端
- Ubuntu 22.04 或更高版本
- Python 3.8+
- PyTorch 2.0+

### 客户端
- 支持 WebSocket 的现代浏览器（Chrome、Firefox、Edge 等）
- 支持麦克风访问

## 安装步骤

### 1. 安装依赖

```bash
cd wenet/runtime/silero_vad_web
pip install -r requirements.txt
```

### 2. 安装 WeNet

```bash
# 从源码安装
cd ../../
pip install -e .
```

或者使用 pip 安装：

```bash
pip install wenet
```

## 使用方法

### 启动服务端

```bash
python app.py
```

可选参数：

```bash
python app.py --host 0.0.0.0 --port 5000 --model firered --device cpu
```

参数说明：
- `--host`: 服务器监听地址（默认：0.0.0.0）
- `--port`: 服务器端口（默认：5000）
- `--model`: WeNet 模型名称（默认：firered，可选：wenetspeech, paraformer, whisper* 等）
- `--device`: 推理设备（默认：cpu，可选：cuda, npu）
- `--debug`: 启用调试模式

### 访问网页界面

启动服务后，在浏览器中访问：

```
http://localhost:5000
```

## 功能说明

### 控制面板

- **开始录音**：点击开始录音，启动新的识别会话
- **停止录音**：点击停止录音，结束当前会话
- **状态指示器**：显示当前连接状态和录音状态

### VAD 参数设置

支持以下参数调节：

1. **检测阈值 (threshold)**
   - 范围：0.0 - 1.0
   - 默认：0.5
   - 说明：语音检测的置信度阈值，值越高检测越严格

2. **最小静音时长 (min_silence_duration_ms)**
   - 范围：50 - 1000 毫秒
   - 默认：100 毫秒
   - 说明：判定语音片段结束所需的最小静音时长

3. **语音填充时长 (speech_pad_ms)**
   - 范围：0 - 500 毫秒
   - 默认：30 毫秒
   - 说明：语音片段前后的填充时长，用于避免截断

### 识别结果

- 实时显示转写文本
- 每个语音片段识别完成后立即更新
- 支持滚动查看历史结果

### 运行日志

- 显示系统运行状态
- 记录识别事件和错误信息
- 包含时间戳便于调试

## 技术架构

### 服务端架构

```
┌─────────────────────────────────────────────────────────────┐
│                      Flask + SocketIO                         │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  WebSocket   │    │  Silero VAD  │    │   WeNet ASR  │  │
│  │   通信层      │    │  语音检测     │    │  语音识别     │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                   │                   │             │
│         └───────────────────┼───────────────────┘             │
│                             ▼                                 │
│                    ┌──────────────┐                           │
│                    │  会话管理    │                           │
│                    │  (Session)   │                           │
│                    └──────────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

### 数据流

1. **音频采集**：浏览器通过 Web Audio API 采集麦克风音频
2. **音频处理**：将音频转换为 16kHz 采样率的 PCM 格式
3. **WebSocket 传输**：通过 WebSocket 实时传输音频数据到服务端
4. **VAD 检测**：Silero VAD 检测语音活动，分割语音片段
5. **语音识别**：WeNet 对每个语音片段进行识别
6. **结果返回**：识别结果通过 WebSocket 实时返回给前端
7. **界面展示**：前端实时更新识别结果和状态

## API 参考

### WebSocket 事件

#### 客户端发送事件

1. **start_session** - 启动新的识别会话
   ```json
   {
     "vad_params": {
       "threshold": 0.5,
       "min_silence_duration_ms": 100,
       "speech_pad_ms": 30
     }
   }
   ```

2. **audio_data** - 发送音频数据
   ```json
   {
     "session_id": "uuid",
     "audio_data": ArrayBuffer
   }
   ```

3. **update_vad_params** - 更新 VAD 参数
   ```json
   {
     "session_id": "uuid",
     "vad_params": {
       "threshold": 0.6
     }
   }
   ```

4. **stop_session** - 停止会话
   ```json
   {
     "session_id": "uuid"
   }
   ```

#### 服务端发送事件

1. **connected** - 连接确认
   ```json
   {
     "status": "connected"
   }
   ```

2. **session_started** - 会话已创建
   ```json
   {
     "session_id": "uuid",
     "vad_params": {...}
   }
   ```

3. **speech_start** - 检测到语音开始
   ```json
   {
     "session_id": "uuid",
     "timestamp": 1.23
   }
   ```

4. **speech_end** - 语音片段结束
   ```json
   {
     "session_id": "uuid",
     "timestamp": 2.45,
     "text": "识别结果"
   }
   ```

5. **recognition_result** - 识别结果
   ```json
   {
     "session_id": "uuid",
     "text": "识别结果",
     "is_final": true
   }
   ```

6. **vad_params_updated** - 参数已更新
   ```json
   {
     "session_id": "uuid",
     "vad_params": {...}
   }
   ```

7. **session_stopped** - 会话已停止
   ```json
   {
     "session_id": "uuid"
   }
   ```

8. **error** - 错误信息
   ```json
   {
     "message": "错误描述"
   }
   ```

## 性能优化建议

### 1. 使用 GPU 加速

如果有 GPU 可用，使用 CUDA 设备可以显著提高识别速度：

```bash
python app.py --device cuda
```

### 2. 调整 VAD 参数

根据实际使用场景调整 VAD 参数：

- **安静环境**：可以降低阈值（如 0.3-0.4）
- **嘈杂环境**：可以提高阈值（如 0.6-0.7）
- **长语音**：增加最小静音时长（如 200-300ms）
- **短语音**：减少最小静音时长（如 50-100ms）

### 3. 网络优化

- 服务端和客户端尽量在同一局域网内
- 使用有线网络连接
- 确保网络延迟稳定

## 常见问题

### Q1: 浏览器无法访问麦克风？

A: 请确保：
- 浏览器已授权麦克风访问权限
- 使用 HTTPS 或 localhost 访问（浏览器安全策略要求）
- 麦克风设备正常工作

### Q2: 识别准确率低？

A: 可以尝试：
- 调整 VAD 阈值，确保完整捕获语音
- 使用更适合的 WeNet 模型
- 确保录音环境安静
- 检查麦克风质量

### Q3: 服务端启动失败？

A: 请检查：
- Python 版本是否符合要求
- 所有依赖是否正确安装
- PyTorch 是否正确安装
- 端口是否被占用

### Q4: 如何更换识别模型？

A: 使用 `--model` 参数指定模型：

```bash
# 使用中文模型
python app.py --model wenetspeech

# 使用多语言模型
python app.py --model whisper-large-v3-turbo
```

## 相关链接

- [Silero VAD GitHub](https://github.com/snakers4/silero-vad)
- [WeNet GitHub](https://github.com/wenet-e2e/wenet)
- [Flask-SocketIO 文档](https://flask-socketio.readthedocs.io/)
- [Web Audio API 文档](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API)

## 许可证

本项目遵循 Apache 2.0 许可证，详见 [LICENSE](../../LICENSE) 文件。
