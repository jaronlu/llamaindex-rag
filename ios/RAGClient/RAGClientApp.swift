import SwiftUI

/// App 入口：目前只有设置页，后续的问答页会挂在这个入口下。
@main
struct RAGClientApp: App {
    @State private var settings = ServerSettings()

    var body: some Scene {
        WindowGroup {
            SettingsView(settings: settings)
        }
    }
}
