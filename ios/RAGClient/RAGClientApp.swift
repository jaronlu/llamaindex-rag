import SwiftUI

/// App 入口：问答页是默认标签，设置页负责服务器地址。
@main
struct RAGClientApp: App {
    @State private var settings = ServerSettings()

    var body: some Scene {
        WindowGroup {
            TabView {
                AskView(settings: settings)
                    .tabItem {
                        Label("问答", systemImage: "text.bubble")
                    }
                SettingsView(settings: settings)
                    .tabItem {
                        Label("设置", systemImage: "gearshape")
                    }
            }
        }
    }
}
