import Foundation

// MARK: - 响应模型

/// 健康检查返回的信息。
///
/// 服务端在索引不可用时仍返回 200，只把 `status` 标成 `unavailable`；客户端必须
/// 区分“连不上服务”和“服务在但索引坏了”，否则会把索引问题误报成网络问题。
struct HealthInfo: Decodable {
    let status: String
    let index: String?
    let chunkCount: Int?
    let model: String?
    let authRequired: Bool?
    let error: String?

    var isReady: Bool { status == "ok" }
}

/// 一条可回到 PDF 核对的证据。
///
/// 字段与服务端 `_serialize_result` 一一对应。页码使用 PDF 物理页码，`pageLabel`
/// 由服务端压缩成 `291–292` 这样的区间；客户端只负责展示，不重新计算，
/// 避免两边对“跨页”给出不同结果。
struct EvidenceItem: Decodable, Identifiable {
    let rank: Int
    let score: Double?
    let rerankScore: Double?
    let originalRank: Int?
    let nodeId: String
    let text: String
    let fileName: String
    let sourcePath: String
    let pageNumbers: [Int]
    let pageLabel: String
    let sourceExists: Bool
    let provenanceComplete: Bool
    let provenanceWarning: String?
    let sourceStatus: String

    var id: String { nodeId }

    /// 把服务端的页码标签转成展示文案；`无` 表示索引里没有页码信息。
    var pageText: String {
        let label = pageLabel.trimmingCharacters(in: .whitespaces)
        return label.isEmpty || label == "无" ? "页码未知" : "第 \(label) 页"
    }
}

/// `done` 事件和一次性 JSON 共用的完整回答。
struct AnswerPayload: Decodable {
    let question: String
    let answer: String
    let citations: [Int]
    let invalidCitations: [Int]
    let llmCalled: Bool
    let evidence: [EvidenceItem]
}

/// 流式回答中的一个事件。
enum AskEvent {
    case status(stage: String, message: String?)
    case evidence([EvidenceItem])
    case delta(String)
    case done(AnswerPayload)
    case error(String)
}

private struct StatusBody: Decodable {
    let stage: String
    let message: String?
}

private struct EvidenceBody: Decodable {
    let items: [EvidenceItem]
}

private struct DeltaBody: Decodable {
    let text: String
}

private struct ErrorBody: Decodable {
    let detail: String
}

// MARK: - SSE 解析

/// 增量解析 Server-Sent Events 帧。
///
/// `URLSession.AsyncBytes.lines` 只保证按行交付，不保证一帧落在一次读取里，因此
/// 解析器自己保存 `event` 和 `data`，遇到空行才派发。多行 `data` 按规范用换行
/// 拼接；注释行（以冒号开头）直接忽略。
struct SSEParser {
    private var eventName: String?
    private var dataLines: [String] = []
    private let decoder: JSONDecoder

    init() {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        self.decoder = decoder
    }

    /// 送入一行；只有遇到空行且缓冲区有数据时才返回事件。
    mutating func consume(line rawLine: String) throws -> AskEvent? {
        let line = rawLine.hasSuffix("\r") ? String(rawLine.dropLast()) : rawLine
        if line.isEmpty {
            return try dispatch()
        }
        if line.hasPrefix(":") {
            return nil
        }
        guard let colon = line.firstIndex(of: ":") else {
            return nil
        }
        let field = String(line[..<colon])
        var value = String(line[line.index(after: colon)...])
        if value.hasPrefix(" ") {
            value.removeFirst()
        }
        switch field {
        case "event":
            eventName = value
        case "data":
            dataLines.append(value)
        default:
            break
        }
        return nil
    }

    /// 流结束时补发最后一个没有空行结尾的事件。
    mutating func finish() throws -> AskEvent? {
        try dispatch()
    }

    private mutating func dispatch() throws -> AskEvent? {
        defer {
            eventName = nil
            dataLines.removeAll(keepingCapacity: true)
        }
        guard !dataLines.isEmpty else {
            return nil
        }
        let name = eventName ?? "message"
        let data = Data(dataLines.joined(separator: "\n").utf8)
        do {
            switch name {
            case "status":
                let body = try decoder.decode(StatusBody.self, from: data)
                return .status(stage: body.stage, message: body.message)
            case "evidence":
                return .evidence(try decoder.decode(EvidenceBody.self, from: data).items)
            case "delta":
                return .delta(try decoder.decode(DeltaBody.self, from: data).text)
            case "done":
                return .done(try decoder.decode(AnswerPayload.self, from: data))
            case "error":
                return .error(try decoder.decode(ErrorBody.self, from: data).detail)
            default:
                // 未知事件类型忽略，便于服务端以后追加事件而不破坏旧客户端。
                return nil
            }
        } catch {
            throw APIClientError.decoding
        }
    }
}

// MARK: - 客户端

/// 与局域网服务通信的最小客户端。
///
/// 只负责拼地址、编码请求、解析 SSE 和把网络错误翻译成中文提示；检索和生成规则
/// 全部在 Mac 上，客户端不解析文档、不保存密钥。
struct APIClient: Sendable {
    let baseURL: URL
    var timeout: TimeInterval = 15
    /// 生成阶段可能持续十几秒，超时只用于发现真正的连接中断。
    var streamTimeout: TimeInterval = 120

    /// 请求 `GET /health`，用于设置页确认地址是否可用。
    func health() async throws -> HealthInfo {
        var request = URLRequest(url: baseURL.appending(path: "health"))
        request.httpMethod = "GET"
        request.timeoutInterval = timeout

        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            guard let http = response as? HTTPURLResponse else {
                throw APIClientError.badResponse
            }
            guard (200..<300).contains(http.statusCode) else {
                throw APIClientError.httpStatus(http.statusCode, detail: Self.detail(from: data))
            }
            return try Self.makeDecoder().decode(HealthInfo.self, from: data)
        } catch let error as APIClientError {
            throw error
        } catch let error as URLError {
            throw APIClientError.network(error)
        } catch {
            throw APIClientError.decoding
        }
    }

    /// 以流式方式调用 `POST /ask`，逐事件交给调用方渲染。
    ///
    /// 返回值是冷流：开始迭代才发请求，取消迭代会取消底层 Task 和 URLSession 请求。
    /// 非 2xx 响应在开始解析前抛出，仍然走普通 HTTP 错误分支；流开始后的失败由服务端
    /// 发送 `error` 事件，客户端不会把它误认为完整回答。
    func streamAnswer(question: String) -> AsyncThrowingStream<AskEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let (bytes, response) = try await URLSession.shared.bytes(
                        for: try makeAskRequest(question: question)
                    )
                    guard let http = response as? HTTPURLResponse else {
                        throw APIClientError.badResponse
                    }
                    guard (200..<300).contains(http.statusCode) else {
                        let data = try await Self.collect(bytes: bytes, limit: 8192)
                        throw APIClientError.httpStatus(
                            http.statusCode,
                            detail: Self.detail(from: data)
                        )
                    }

                    // 不能用 `bytes.lines`：部分系统版本会丢掉空行，而空行正是 SSE
                    // 的事件分隔符，丢掉后所有事件会被拼成一帧。这里按字节自己切行，
                    // 保留空行；跨读取的半个事件由 SSEParser 缓冲。
                    var parser = SSEParser()
                    var lineBytes: [UInt8] = []
                    for try await byte in bytes {
                        if byte == 0x0A {
                            let line = String(decoding: lineBytes, as: UTF8.self)
                            lineBytes.removeAll(keepingCapacity: true)
                            if let event = try parser.consume(line: line) {
                                continuation.yield(event)
                            }
                        } else {
                            lineBytes.append(byte)
                        }
                    }
                    if !lineBytes.isEmpty {
                        let line = String(decoding: lineBytes, as: UTF8.self)
                        if let event = try parser.consume(line: line) {
                            continuation.yield(event)
                        }
                    }
                    if let event = try parser.finish() {
                        continuation.yield(event)
                    }
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish()
                } catch let error as URLError where error.code == .cancelled {
                    continuation.finish()
                } catch let error as APIClientError {
                    continuation.finish(throwing: error)
                } catch let error as URLError {
                    continuation.finish(throwing: APIClientError.network(error))
                } catch {
                    continuation.finish(throwing: APIClientError.decoding)
                }
            }
            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }

    private func makeAskRequest(question: String) throws -> URLRequest {
        struct AskBody: Encodable {
            let question: String
            let stream: Bool
        }

        var request = URLRequest(url: baseURL.appending(path: "ask"))
        request.httpMethod = "POST"
        request.timeoutInterval = streamTimeout
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("text/event-stream", forHTTPHeaderField: "Accept")
        do {
            request.httpBody = try JSONEncoder().encode(AskBody(question: question, stream: true))
        } catch {
            throw APIClientError.invalidRequest
        }
        return request
    }

    /// 只读取有限的错误响应体，避免异常服务端返回超大内容拖住界面。
    private static func collect(bytes: URLSession.AsyncBytes, limit: Int) async throws -> Data {
        var data = Data()
        for try await byte in bytes {
            data.append(byte)
            if data.count >= limit {
                break
            }
        }
        return data
    }

    private static func makeDecoder() -> JSONDecoder {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return decoder
    }

    /// 提取服务端的 `detail` 字段；没有时返回 nil，不把原始响应体直接展示给用户。
    private static func detail(from data: Data) -> String? {
        try? JSONDecoder().decode(ErrorBody.self, from: data).detail
    }
}

enum APIClientError: LocalizedError {
    case network(URLError)
    case badResponse
    case httpStatus(Int, detail: String?)
    case decoding
    case invalidRequest

    var errorDescription: String? {
        switch self {
        case .network(let error):
            switch error.code {
            case .cannotConnectToHost, .cannotFindHost, .networkConnectionLost:
                return "连接不上服务器，请确认 Mac 上的服务已启动，并使用 --host 0.0.0.0"
            case .timedOut:
                return "请求超时，可能是索引或模型加载较慢，稍后重试"
            default:
                return "网络错误：\(error.localizedDescription)"
            }
        case .badResponse:
            return "服务返回了无法识别的响应"
        case .httpStatus(let code, let detail):
            if let detail, !detail.isEmpty {
                return "服务返回 \(code)：\(detail)"
            }
            return "服务返回状态码 \(code)"
        case .decoding:
            return "服务返回的 JSON 格式与客户端预期不一致"
        case .invalidRequest:
            return "无法构造请求"
        }
    }
}
