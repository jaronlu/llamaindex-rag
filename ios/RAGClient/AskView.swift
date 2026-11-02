import SwiftUI
import UIKit

/// 问答页：输入问题，先看证据，再看流式回答。
///
/// 页面不持有任何文档或密钥，只把问题发给 Mac 上的服务。证据在生成开始前就会到达，
/// 因此即使生成很慢或失败，用户仍能先核对来源；只有收到 `done` 才显示为完整回答。
struct AskView: View {
    let settings: ServerSettings
    @State private var model = AskModel()

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    if model.phase == .idle {
                        intro
                    }
                    status
                    if !model.answerText.isEmpty {
                        answerSection
                    }
                    citationNotice
                    if !model.evidence.isEmpty {
                        evidenceSection
                    }
                    if let error = model.errorMessage {
                        errorView(error)
                    }
                }
                .padding()
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .scrollDismissesKeyboard(.interactively)
            .navigationTitle("问答")
            .safeAreaInset(edge: .bottom) {
                inputBar
            }
        }
    }

    private var intro: some View {
        VStack(alignment: .leading, spacing: 8) {
            Image(systemName: "text.magnifyingglass")
                .font(.largeTitle)
                .foregroundStyle(.tint)
            Text("检索、重排和生成都在 Mac 上完成。")
                .font(.headline)
            Text("提问后先显示证据，再逐段显示回答。每条证据都带文件名和 PDF 物理页码，可打开原文件核对。生成会调用云端模型并产生费用。")
                .font(.callout)
                .foregroundStyle(.secondary)
        }
        .padding(.top, 8)
    }

    @ViewBuilder
    private var status: some View {
        switch model.phase {
        case .retrieving, .generating:
            HStack(spacing: 8) {
                ProgressView()
                    .controlSize(.small)
                Text(model.statusMessage.isEmpty ? "处理中…" : model.statusMessage)
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
        case .idle, .completed, .failed:
            EmptyView()
        }
    }

    private var answerSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("回答")
                .font(.headline)
            Text(model.answerText)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    @ViewBuilder
    private var citationNotice: some View {
        if model.phase == .completed {
            if !model.invalidCitations.isEmpty {
                Label(
                    "回答引用了不存在的证据编号：\(citationNumbers(model.invalidCitations))",
                    systemImage: "exclamationmark.triangle"
                )
                .font(.callout)
                .foregroundStyle(.orange)
            } else if model.llmCalled && model.citations.isEmpty {
                Label("回答没有引用任何证据，请打开原文核对", systemImage: "exclamationmark.triangle")
                    .font(.callout)
                    .foregroundStyle(.orange)
            } else if !model.citations.isEmpty {
                Label("引用证据：\(citationNumbers(model.citations))", systemImage: "checkmark.circle")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var evidenceSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("证据（\(model.evidence.count)）")
                .font(.headline)
            ForEach(model.evidence) { item in
                EvidenceRow(item: item)
            }
        }
    }

    private func errorView(_ message: String) -> some View {
        Label(message, systemImage: "xmark.octagon")
            .font(.callout)
            .foregroundStyle(.red)
    }

    private var inputBar: some View {
        HStack(alignment: .bottom, spacing: 8) {
            TextField("输入问题，例如：年休假有多少天？", text: $model.question, axis: .vertical)
                .lineLimit(1...4)
                .padding(.horizontal, 12)
                .padding(.vertical, 10)
                .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 18))
                .disabled(model.isRunning)
                .submitLabel(.send)

            Button {
                if model.isRunning {
                    model.cancel()
                } else {
                    model.ask(using: settings)
                }
            } label: {
                Image(systemName: model.isRunning ? "stop.circle.fill" : "arrow.up.circle.fill")
                    .font(.system(size: 32))
                    .symbolRenderingMode(.hierarchical)
            }
            .disabled(!model.isRunning && model.question.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            .accessibilityLabel(model.isRunning ? "停止生成" : "发送问题")
        }
        .padding(.horizontal)
        .padding(.vertical, 8)
        .background(.bar)
    }

    private func citationNumbers(_ values: [Int]) -> String {
        values.map { "[\($0)]" }.joined(separator: "、")
    }
}

/// 一条证据：折叠时显示来源和页码，展开后显示完整原文和本机路径。
private struct EvidenceRow: View {
    let item: EvidenceItem
    @State private var expanded = false

    var body: some View {
        DisclosureGroup(isExpanded: $expanded) {
            VStack(alignment: .leading, spacing: 10) {
                Text(item.text)
                    .font(.callout)
                    .textSelection(.enabled)
                if !item.provenanceComplete, let warning = item.provenanceWarning {
                    Text(warning)
                        .font(.caption)
                        .foregroundStyle(.orange)
                }
                Text(item.sourcePath)
                    .font(.caption2.monospaced())
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                Button {
                    UIPasteboard.general.string = item.text
                } label: {
                    Label("复制原文", systemImage: "doc.on.doc")
                        .font(.caption)
                }
                .buttonStyle(.bordered)
            }
            .padding(.top, 8)
        } label: {
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("证据 \(item.rank)")
                        .font(.caption.weight(.semibold))
                    Spacer()
                    Text(item.provenanceComplete ? "来源可核对" : "来源不完整")
                        .font(.caption2)
                        .foregroundStyle(item.provenanceComplete ? Color.secondary : Color.orange)
                }
                Text(item.fileName)
                    .font(.subheadline)
                HStack(spacing: 8) {
                    Text(item.pageText)
                    if let scoreText {
                        Text(scoreText)
                    }
                }
                .font(.caption)
                .foregroundStyle(.secondary)
            }
        }
        .padding(12)
        .background(.quaternary.opacity(0.35), in: RoundedRectangle(cornerRadius: 12))
    }

    private var scoreText: String? {
        if let rerank = item.rerankScore {
            return "重排分数 \(String(format: "%.3f", rerank))"
        }
        if let score = item.score {
            return "相似度 \(String(format: "%.3f", score))"
        }
        return nil
    }
}
