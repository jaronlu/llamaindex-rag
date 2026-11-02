import Foundation
import Observation

/// 问答页的状态与流式请求生命周期。
///
/// 视图只负责展示，这里集中处理“什么时候算失败”和“取消后保留什么”。判断规则与
/// 服务端一致：只有收到 `done` 才算完整回答；`error`、流提前结束或网络错误都
/// 标记为失败，已收到的证据仍然保留，方便对照原文。
@MainActor
@Observable
final class AskModel {
    enum Phase: Equatable {
        case idle
        case retrieving
        case generating
        case completed
        case failed
    }

    var question = ""
    private(set) var phase: Phase = .idle
    private(set) var statusMessage = ""
    private(set) var answerText = ""
    private(set) var evidence: [EvidenceItem] = []
    private(set) var citations: [Int] = []
    private(set) var invalidCitations: [Int] = []
    private(set) var llmCalled = false
    private(set) var errorMessage: String?

    @ObservationIgnored private var task: Task<Void, Never>?

    var isRunning: Bool {
        phase == .retrieving || phase == .generating
    }

    /// 发送当前问题。重复调用会先取消上一次请求，避免两个响应交错写入同一份状态。
    func ask(using settings: ServerSettings) {
        let trimmed = question.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            fail("请先输入问题")
            return
        }

        let baseURL: URL
        do {
            baseURL = try settings.resolvedBaseURL()
        } catch {
            fail(error.localizedDescription)
            return
        }

        task?.cancel()
        reset()
        phase = .retrieving
        statusMessage = "正在检索证据…"

        let client = APIClient(baseURL: baseURL)
        task = Task { [weak self] in
            guard let self else { return }
            do {
                for try await event in client.streamAnswer(question: trimmed) {
                    guard !Task.isCancelled else { return }
                    handle(event)
                }
                guard !Task.isCancelled else { return }
                if phase != .completed && phase != .failed {
                    fail("连接中断，未收到完整回答；已展示的证据仍可核对")
                }
            } catch {
                guard !Task.isCancelled else { return }
                fail(error.localizedDescription)
            }
        }
    }

    /// 取消正在进行的请求，并明确告诉用户这不是完整结果。
    func cancel() {
        task?.cancel()
        task = nil
        guard isRunning else { return }
        phase = .failed
        statusMessage = ""
        errorMessage = "已取消本次提问，未收到完整回答"
    }

    private func reset() {
        answerText = ""
        evidence = []
        citations = []
        invalidCitations = []
        llmCalled = false
        errorMessage = nil
        statusMessage = ""
    }

    private func handle(_ event: AskEvent) {
        switch event {
        case .status(let stage, let message):
            switch stage {
            case "retrieving":
                phase = .retrieving
            case "generating":
                phase = .generating
            default:
                break
            }
            if let message, !message.isEmpty {
                statusMessage = message
            }
        case .evidence(let items):
            evidence = items
        case .delta(let text):
            if phase != .generating {
                phase = .generating
            }
            answerText += text
        case .done(let payload):
            // 以 done 的完整文本为准，避免流式拼接与最终结果出现细微差异。
            answerText = payload.answer
            evidence = payload.evidence
            citations = payload.citations
            invalidCitations = payload.invalidCitations
            llmCalled = payload.llmCalled
            phase = .completed
            statusMessage = ""
        case .error(let detail):
            fail(detail)
        }
    }

    private func fail(_ message: String) {
        phase = .failed
        statusMessage = ""
        errorMessage = message
    }
}
