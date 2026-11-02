import Foundation
import Observation

/// 保存并校验服务端地址。
///
/// 地址只存在 UserDefaults：客户端不保存索引、文档或任何密钥，换一个地址就能换一台
/// Mac。校验集中在这里，避免设置页和请求层对“什么算合法地址”给出不同判断。
@Observable
final class ServerSettings {
    /// 用户输入框里的原文，保留原文是为了在输入到一半时不打断编辑。
    var baseURLText: String {
        didSet { defaults.set(baseURLText, forKey: Self.key) }
    }

    private static let key = "server.baseURL"
    private let defaults: UserDefaults

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        self.baseURLText = defaults.string(forKey: Self.key) ?? "http://127.0.0.1:8000"
    }

    /// 把输入转换成请求用的地址。
    ///
    /// 只接受 http/https，并且必须带主机名；端口缺省时由 URL 自己处理。这里不做
    /// 域名白名单，因为服务本来就跑在局域网内，地址由使用者自己填写。
    func resolvedBaseURL() throws -> URL {
        let trimmed = baseURLText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw ServerAddressError.empty
        }
        guard let url = URL(string: trimmed), let scheme = url.scheme?.lowercased(),
              scheme == "http" || scheme == "https" else {
            throw ServerAddressError.invalid
        }
        guard let host = url.host, !host.isEmpty else {
            throw ServerAddressError.invalid
        }
        return url
    }
}

enum ServerAddressError: LocalizedError {
    case empty
    case invalid

    var errorDescription: String? {
        switch self {
        case .empty:
            return "请填写服务器地址，例如 http://192.168.1.10:8000"
        case .invalid:
            return "地址格式不正确，需要以 http:// 或 https:// 开头并包含主机名"
        }
    }
}
