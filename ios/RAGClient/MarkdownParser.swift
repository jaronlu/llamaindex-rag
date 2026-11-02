import Foundation

/// 回答里支持的最小 Markdown 块级结构。
///
/// 只覆盖生成模型常用的语法，避免为了几个符号引入第三方渲染库。行内样式（加粗、
/// 斜体、行内代码、链接）不在这里处理，交给系统 Markdown 解析器。
enum MarkdownBlock {
    case heading(level: Int, text: String)
    case paragraph(String)
    case bullets([String])
    case numbered([(number: Int, text: String)])
    case quote(String)
    case code(String)
    case divider
}

/// 把文本切成可独立排版的块。
///
/// 流式输出会反复调用：每次拿到新的 `delta` 都重新解析全文，因此解析器必须对
/// 不完整输入保持宽容，不能因为未闭合的代码块或加粗符号丢掉已有内容。
enum MarkdownParser {
    static func blocks(from text: String) -> [MarkdownBlock] {
        var blocks: [MarkdownBlock] = []
        var paragraph: [String] = []
        var bullets: [String] = []
        var numbered: [(number: Int, text: String)] = []
        var codeLines: [String] = []
        var inCode = false

        func flushParagraph() {
            guard !paragraph.isEmpty else { return }
            blocks.append(.paragraph(paragraph.joined(separator: "\n")))
            paragraph.removeAll()
        }

        func flushLists() {
            if !bullets.isEmpty {
                blocks.append(.bullets(bullets))
                bullets.removeAll()
            }
            if !numbered.isEmpty {
                blocks.append(.numbered(numbered))
                numbered.removeAll()
            }
        }

        func flushAll() {
            flushParagraph()
            flushLists()
        }

        for rawLine in text.components(separatedBy: "\n") {
            let line = rawLine.trimmingCharacters(in: .whitespaces)

            if inCode {
                if line.hasPrefix("```") {
                    blocks.append(.code(codeLines.joined(separator: "\n")))
                    codeLines.removeAll()
                    inCode = false
                } else {
                    codeLines.append(rawLine)
                }
                continue
            }

            if line.hasPrefix("```") {
                flushAll()
                inCode = true
                continue
            }

            if line.isEmpty {
                flushAll()
                continue
            }

            if isDivider(line) {
                flushAll()
                blocks.append(.divider)
                continue
            }

            if let (level, heading) = heading(line) {
                flushAll()
                blocks.append(.heading(level: level, text: heading))
                continue
            }

            if let item = bulletItem(line) {
                flushParagraph()
                if !numbered.isEmpty {
                    blocks.append(.numbered(numbered))
                    numbered.removeAll()
                }
                bullets.append(item)
                continue
            }

            if let (number, item) = numberedItem(line) {
                flushParagraph()
                if !bullets.isEmpty {
                    blocks.append(.bullets(bullets))
                    bullets.removeAll()
                }
                numbered.append((number, item))
                continue
            }

            if let quote = quoteLine(line) {
                flushAll()
                blocks.append(.quote(quote))
                continue
            }

            // 列表项偶尔被模型换行；续行归到上一项，不单独成为段落。
            if !bullets.isEmpty {
                bullets[bullets.count - 1] += "\n" + line
                continue
            }
            if !numbered.isEmpty {
                numbered[numbered.count - 1].text += "\n" + line
                continue
            }
            paragraph.append(line)
        }

        // 流式过程中代码块可能还没闭合，已有内容仍然展示。
        if inCode {
            blocks.append(.code(codeLines.joined(separator: "\n")))
        }
        flushAll()
        return blocks
    }

    private static func heading(_ line: String) -> (Int, String)? {
        let hashes = line.prefix { $0 == "#" }
        guard (1...6).contains(hashes.count) else { return nil }
        let rest = line.dropFirst(hashes.count)
        guard rest.first == " " else { return nil }
        return (hashes.count, String(rest).trimmingCharacters(in: .whitespaces))
    }

    private static func bulletItem(_ line: String) -> String? {
        for marker in ["- ", "* ", "+ "] where line.hasPrefix(marker) {
            return String(line.dropFirst(marker.count))
        }
        return nil
    }

    private static func numberedItem(_ line: String) -> (Int, String)? {
        guard let space = line.firstIndex(of: " ") else { return nil }
        let token = line[..<space]
        guard token.hasSuffix(".") else { return nil }
        let digits = token.dropLast()
        guard !digits.isEmpty, digits.allSatisfy({ $0.isNumber }) else { return nil }
        let text = line[line.index(after: space)...]
        return (Int(digits) ?? 1, String(text))
    }

    private static func quoteLine(_ line: String) -> String? {
        if line.hasPrefix("> ") {
            return String(line.dropFirst(2))
        }
        if line == ">" {
            return ""
        }
        return nil
    }

    private static func isDivider(_ line: String) -> Bool {
        guard line.count >= 3, let first = line.first, "-*_".contains(first) else {
            return false
        }
        return line.allSatisfy { $0 == first }
    }
}
