//
//  ContentView.swift
//  DriverAssist
//
//  Created by Rick Clark on 7/20/26.
//

import SwiftUI

@MainActor
struct ContentView: View {
    @StateObject private var modelManager = ModelManager(defaultModel: .small)

    var body: some View {
        VStack {
            Image(systemName: "globe")
                .imageScale(.large)
                .foregroundStyle(.tint)
            Text("Hello, world!")
        }
        .onAppear {
            modelManager.loadInitialModel()
        }
        .padding()
    }
}

#Preview {
    ContentView()
}
