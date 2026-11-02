# iPhone 客户端

SwiftUI 客户端，只通过局域网调用 Mac 上的 `rag.cli.serve`。App 内没有索引、文档、嵌入模型或生成密钥。

当前进度：问答页（流式回答、证据和引用检查）+ 设置页（地址与 `GET /health` 连通性检查）。

## 目录

```text
ios/
├── RAGClient.xcodeproj/          # 手写的 Xcode 工程（同步文件夹，无需 xcodegen）
├── RAGClient-Info.plist          # ATS 例外与本地网络权限说明
└── RAGClient/
    ├── RAGClientApp.swift        # 问答 / 设置两个标签
    ├── AskView.swift             # 提问、流式回答、证据列表
    ├── AskModel.swift            # 问题状态与请求取消
    ├── SettingsView.swift
    ├── ServerSettings.swift      # 地址校验与 UserDefaults 持久化
    └── APIClient.swift           # /health、/ask 流式请求和 SSE 解析
```

## 运行（模拟器）

1. 在项目根目录启动服务，模拟器与 Mac 共享网络，用默认监听即可：

   ```bash
   uv run python -m src.rag.cli.serve data/quickstart-index
   ```

2. 打开 `ios/RAGClient.xcodeproj`，选择 iPhone 模拟器，运行（⌘R）。

3. 设置页默认地址是 `http://127.0.0.1:8000`，点“测试连接”。成功时显示状态、片段数、模型和索引路径；索引损坏时状态显示“索引不可用”并给出原因。

最低 iOS 版本 18.0。

## 问答流程

1. 切到“问答”标签，输入问题后点发送。请求固定使用 `POST /ask` 和 `"stream": true`。
2. 界面先显示“正在检索证据”，收到 `evidence` 后立刻列出文件名、PDF 物理页码和来源状态；点开可看完整原文和本机路径，并复制原文。
3. 回答随 `delta` 逐段出现；收到 `done` 后才显示引用检查（引用编号、无效编号、未引用提示）。
4. 生成中可点停止按钮取消；取消、流中断和 `error` 都会明确标记为不完整，已收到的证据保留。

生成会调用 Mac 上配置的云端模型并产生费用；检索和证据展示本身不调用生成模型。

## 运行（自己的 iPhone）

1. 服务改为监听局域网：

   ```bash
   uv run python -m src.rag.cli.serve data/quickstart-index --host 0.0.0.0
   ipconfig getifaddr en0   # Mac 的局域网 IP
   ```

2. Xcode 的 Target → Signing & Capabilities 选择自己的 Team（免费 Apple ID 即可，安装有效 7 天），Bundle Identifier 改成唯一值。

3. 运行到真机，首次会弹出本地网络权限请求，允许后再点“测试连接”。地址填 `http://<Mac IP>:8000`。

## 安全边界

- `RAGClient-Info.plist` 设置了 `NSAllowsArbitraryLoads`，因为局域网地址通常是明文 HTTP 且 IP 不固定。这只用于本地开发，不要把地址指向公网。
- 暂不接入 `RAG_API_TOKEN`；服务端默认不鉴权，仅限可信局域网。
- 客户端不缓存文档或索引，只把服务器地址存进 UserDefaults。
- 真机与 Mac 必须处于同一 Wi-Fi/局域网；公司网络或访客网络可能隔离设备。

## 已知限制

- 手写工程使用 Xcode 16+ 的同步文件夹；改动 `RAGClient/` 下的文件不需要动工程文件，新增目录同理。
- 没有单元测试目标；逻辑验证靠 Xcode 运行和真实服务联调。
- 地址仍需手动填写，App 不广播也不自动发现服务；IP 变化后要重新查并修改。
- 问答页不保存历史记录，切换标签或重新提问不会保留上一次结果。
