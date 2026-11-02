import SwiftUI

/// 设置页：填写服务地址并确认服务是否可用。
///
/// 这一步只验证连通性，不做检索或生成；地址、索引状态和错误原因直接展示，
/// 方便在没有 Xcode 调试的情况下判断是网络、服务还是索引的问题。
struct SettingsView: View {
    @Bindable var settings: ServerSettings
    @State private var state: ConnectionState = .idle

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    TextField("http://192.168.1.10:8000", text: $settings.baseURLText)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .keyboardType(.URL)
                        .onChange(of: settings.baseURLText) {
                            // 地址变了就清掉上一次结果，避免把旧服务器的状态当成本次的。
                            state = .idle
                        }
                } header: {
                    Text("服务器地址")
                } footer: {
                    Text("Mac 上运行 rag.cli.serve 后，填入它打印的地址；手机访问需要以 --host 0.0.0.0 启动。模拟器可直接用 127.0.0.1。")
                }

                Section {
                    Button {
                        Task { await check() }
                    } label: {
                        HStack {
                            Text("测试连接")
                            Spacer()
                            if case .checking = state {
                                ProgressView()
                            }
                        }
                    }
                    .disabled(isChecking)
                }

                resultSection
            }
            .navigationTitle("RAG 客户端")
        }
    }

    @ViewBuilder
    private var resultSection: some View {
        switch state {
        case .idle:
            EmptyView()
        case .checking:
            Section("检查中") {
                Text("正在请求 /health …")
                    .foregroundStyle(.secondary)
            }
        case .success(let health):
            Section("服务状态") {
                LabeledContent("状态", value: health.isReady ? "可用" : "索引不可用")
                if let count = health.chunkCount {
                    LabeledContent("片段数", value: String(count))
                }
                if let model = health.model {
                    LabeledContent("模型", value: model)
                }
                if let index = health.index {
                    LabeledContent("索引路径") {
                        Text(index)
                            .font(.caption.monospaced())
                            .multilineTextAlignment(.trailing)
                    }
                }
                if let error = health.error {
                    Text(error)
                        .font(.callout)
                        .foregroundStyle(.orange)
                }
            }
        case .failure(let message):
            Section("连接失败") {
                Text(message)
                    .foregroundStyle(.red)
            }
        }
    }

    private var isChecking: Bool {
        if case .checking = state { return true }
        return false
    }

    private func check() async {
        let client: APIClient
        do {
            let url = try settings.resolvedBaseURL()
            client = APIClient(baseURL: url)
        } catch {
            state = .failure(error.localizedDescription)
            return
        }

        state = .checking
        do {
            state = .success(try await client.health())
        } catch {
            state = .failure(error.localizedDescription)
        }
    }
}

/// 设置页的三种终态；`checking` 用于展示进度。
enum ConnectionState {
    case idle
    case checking
    case success(HealthInfo)
    case failure(String)
}
