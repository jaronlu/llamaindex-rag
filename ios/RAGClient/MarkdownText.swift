import SwiftUI

/// 把回答文本排成 Markdown 样式。
///
/// 块级结构由 `MarkdownParser` 拆分，行内样式交给系统的 AttributedString：这样
/// 加粗、斜体、行内代码和链接不需要自己写解析器，列表和标题也能按块排版。
/// 流式阶段文本可能暂时不完整，单个块解析失败时按纯文本显示，不会丢内容。
struct MarkdownText: View {
    let text: String

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            ForEach(Array(MarkdownParser.blocks(from: text).enumerated()), id: \.offset) { _, block in
                MarkdownBlockView(block: block)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .textSelection(.enabled)
    }
}

private struct MarkdownBlockView: View {
    let block: MarkdownBlock

    var body: some View {
        switch block {
        case .heading(let level, let text):
            Text(inline(text))
                .font(headingFont(level))
                .fontWeight(.semibold)
        case .paragraph(let text):
            Text(inline(text))
                .frame(maxWidth: .infinity, alignment: .leading)
        case .bullets(let items):
            VStack(alignment: .leading, spacing: 6) {
                ForEach(Array(items.enumerated()), id: \.offset) { _, item in
                    HStack(alignment: .firstTextBaseline, spacing: 8) {
                        Text("•")
                            .foregroundStyle(.secondary)
                        Text(inline(item))
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
            }
        case .numbered(let items):
            VStack(alignment: .leading, spacing: 6) {
                ForEach(Array(items.enumerated()), id: \.offset) { _, item in
                    HStack(alignment: .firstTextBaseline, spacing: 8) {
                        Text("\(item.number).")
                            .foregroundStyle(.secondary)
                            .monospacedDigit()
                        Text(inline(item.text))
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
            }
        case .quote(let text):
            HStack(alignment: .top, spacing: 8) {
                Rectangle()
                    .fill(.secondary)
                    .frame(width: 3)
                Text(inline(text))
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        case .code(let code):
            ScrollView(.horizontal, showsIndicators: false) {
                Text(code)
                    .font(.system(.callout, design: .monospaced))
                    .padding(10)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
        case .divider:
            Divider()
        }
    }

    /// 行内 Markdown；未闭合的标记或非法语法直接退回纯文本。
    private func inline(_ text: String) -> AttributedString {
        var options = AttributedString.MarkdownParsingOptions()
        options.interpretedSyntax = .inlineOnlyPreservingWhitespace
        return (try? AttributedString(markdown: text, options: options)) ?? AttributedString(text)
    }

    private func headingFont(_ level: Int) -> Font {
        switch level {
        case 1:
            return .title2
        case 2:
            return .title3
        case 3:
            return .headline
        default:
            return .subheadline
        }
    }
}
