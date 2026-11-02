import Foundation

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

/// 与局域网服务通信的最小客户端。
///
/// 只负责拼地址、超时和把网络错误翻译成中文提示；检索和生成规则全部在 Mac 上，
/// 客户端不解析文档、不保存密钥。
struct APIClient {
    let baseURL: URL
    var timeout: TimeInterval = 15

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
            let decoder = JSONDecoder()
            decoder.keyDecodingStrategy = .convertFromSnakeCase
            return try decoder.decode(HealthInfo.self, from: data)
        } catch let error as APIClientError {
            throw error
        } catch let error as URLError {
            throw APIClientError.network(error)
        } catch {
            throw APIClientError.decoding
        }
    }

    /// 提取服务端的 `detail` 字段；没有时返回 nil，不把原始响应体直接展示给用户。
    private static func detail(from data: Data) -> String? {
        struct ErrorBody: Decodable { let detail: String? }
        return try? JSONDecoder().decode(ErrorBody.self, from: data).detail
    }
}

enum APIClientError: LocalizedError {
    case network(URLError)
    case badResponse
    case httpStatus(Int, detail: String?)
    case decoding

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
        }
    }
}
