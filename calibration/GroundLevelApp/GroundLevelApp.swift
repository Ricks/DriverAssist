// GroundLevelApp.swift
//
// Measures how far a parking spot is from level, at the iPhone
// inclinometer's own precision. Pitch = fore/aft slope, roll = side slope.
//
// ROSETTE (4x90 deg) -- the only capture mode. Take four 3-second readings
// at one spot, rotating the phone 90 deg CLOCKWISE (viewed from above)
// between each. A least-squares fit over the 8 numbers (4 pitch + 4 roll)
// solves for the phone's fixed inclinometer bias, the true ground-slope
// vector (magnitude + downhill direction), and a residual RMS that says
// how good the four turns were. Four headings beat a single 180 deg flip:
// per-capture noise drops, small turn-angle errors wash out, and a bad
// turn shows up as a large residual instead of silently biasing the
// answer. Reported pitch/roll are in the orientation of capture 1 (top
// edge = the way you started).
//
// SESSIONS. Name a run and every rosette is appended to a JSON file under
// Documents/Sessions/, with per-reading and per-session timestamps.
// Browse / share / delete from the "Sessions" button; each file is a
// standalone .json for offline analysis.
//
// Build: open GroundLevel.xcodeproj and run to a real device (motion isn't
// simulated). No capabilities / usage strings needed for CMDeviceMotion.

import SwiftUI
import CoreMotion
import Darwin

@main
struct GroundLevelApp: App {
    var body: some Scene { WindowGroup { ContentView() } }
}

// MARK: - Motion

struct Capture: Equatable {
    let pitch, pitchSD, roll, rollSD: Double
    let seconds: Double
    let n: Int
}

/// Least-squares fit of the 4x90 deg rosette.
///
/// Model, with the phone rotated CLOCKWISE by i*90 deg from capture 1 and a
/// fixed phone-frame bias (Bp, Br) plus a ground-slope vector (gx, gy)
/// expressed in capture 1's frame (gx = fore/aft, gy = left/right):
///
///   p_i = Bp + gx*cos(i*90) + gy*sin(i*90)
///   r_i = Br - gx*sin(i*90) + gy*cos(i*90)
///
/// which for i = 0,1,2,3 gives the closed form used below.
struct RosetteResult: Equatable {
    let biasPitch, biasRoll: Double
    let truePitch, trueRoll: Double        // capture-1 frame == slope components gx, gy
    let truePitchSD, trueRollSD: Double
    let slopeMagnitude: Double
    let downhillDirDeg: Double              // 0 deg = capture-1 top edge, clockwise positive
    let residualRMS: Double                 // RMS of the 8 fit residuals -- turn quality
    let caps: [Capture]                     // the 4, in capture order
    var suspect: Bool { residualRMS > 0.15 }

    static func solve(_ c: [Capture]) -> RosetteResult {
        let p = c.map { $0.pitch },  r  = c.map { $0.roll }
        let ps = c.map { $0.pitchSD }, rs = c.map { $0.rollSD }

        let bp = (p[0] + p[1] + p[2] + p[3]) / 4
        let br = (r[0] + r[1] + r[2] + r[3]) / 4
        // Each slope component is estimated twice -- once from pitch, once
        // from roll -- and averaged (this IS the least-squares answer for
        // 4 evenly-spaced samples).
        let gx = ((p[0] - p[2]) + (r[3] - r[1])) / 4
        let gy = ((p[1] - p[3]) + (r[0] - r[2])) / 4

        var ss = 0.0
        for i in 0..<4 {
            let a = Double(i) * .pi / 2
            let pp = bp + gx * cos(a) + gy * sin(a)
            let rr = br - gx * sin(a) + gy * cos(a)
            ss += (p[i] - pp) * (p[i] - pp) + (r[i] - rr) * (r[i] - rr)
        }
        let rms = (ss / 8).squareRoot()

        let gxSD = (ps[0]*ps[0] + ps[2]*ps[2] + rs[3]*rs[3] + rs[1]*rs[1]).squareRoot() / 4
        let gySD = (ps[1]*ps[1] + ps[3]*ps[3] + rs[0]*rs[0] + rs[2]*rs[2]).squareRoot() / 4

        var dn = atan2(-gy, -gx) * 180 / .pi
        if dn < 0 { dn += 360 }

        return RosetteResult(biasPitch: bp, biasRoll: br,
                             truePitch: gx, trueRoll: gy,
                             truePitchSD: gxSD, trueRollSD: gySD,
                             slopeMagnitude: (gx*gx + gy*gy).squareRoot(),
                             downhillDirDeg: dn, residualRMS: rms, caps: c)
    }
}

final class Inclinometer: ObservableObject {
    @Published var pitchDeg = 0.0        // + = top edge raised
    @Published var rollDeg  = 0.0        // + = right edge raised
    @Published var tiltDeg  = 0.0
    @Published var attPitchDeg = 0.0
    @Published var attRollDeg  = 0.0
    @Published var pitchSigma = 0.0     // 3 s window — for the readout / indicator
    @Published var rollSigma  = 0.0
    @Published var rateHz     = 0.0
    var steady: Bool { pitchSigma < 0.03 && rollSigma < 0.03 }

    // HYSTERETIC "steady enough to press the button". Plain `steady`
    // flickers true/false as σ hovers around 0.03 right after a turn, and
    // a button that toggles `.disabled` every few frames eats the taps
    // that land in a disabled window. This latches: on once σ is genuinely
    // low, off only once it is clearly moving again.
    @Published var canCapture = false

    // A capture cycle: tap -> `capturing` (+ `waitingForSteady` only if the
    // tap itself nudged σ over the line) -> the 3-second accumulation runs
    // -> finish().
    @Published var capturing = false
    @Published var waitingForSteady = false
    @Published var armTimedOut = false
    @Published var captureRemaining = 0.0

    enum RosetteStep: Equatable { case idle, have1, have2, have3, done }
    @Published var rosStep: RosetteStep = .idle
    @Published var rosCaptures: [Capture] = []
    @Published var rosResult: RosetteResult?

    // Queue of completed readings not yet written to a session. A QUEUE
    // (not a flag + "latest result") so a reading can't be lost if two
    // finish before ContentView drains -- it drains the whole array.
    enum ReadingPayload: Equatable { case rosette(RosetteResult) }
    @Published var readingQueue: [ReadingPayload] = []

    // On-screen breadcrumb log for diagnosing the capture flow.
    @Published var log: [String] = []
    func note(_ s: String) {
        let t = Date().timeIntervalSince1970.truncatingRemainder(dividingBy: 100)
        log.append(String(format: "%05.1f %@", t, s))
        if log.count > 9 { log.removeFirst() }
    }

    func clearPending() {
        readingQueue.removeAll()
        capturing = false; waitingForSteady = false; captureRemaining = 0
        rosReset()
    }

    private var captureSeconds = 3.0
    private var capStart = Date()
    private var capArmedAt = Date()
    private let armTimeout = 45.0
    private var capP: [Double] = []
    private var capR: [Double] = []

    private let mm = CMMotionManager()
    private let queue: OperationQueue = {
        let q = OperationQueue(); q.maxConcurrentOperationCount = 1; q.name = "inclinometer"; return q
    }()
    private let bufCap = 300
    private var pBuf: [Double] = []
    private var rBuf: [Double] = []
    private var lastStamp: TimeInterval = 0

    func start() {
        guard mm.isDeviceMotionAvailable else { return }
        mm.deviceMotionUpdateInterval = 1.0 / 100.0
        mm.startDeviceMotionUpdates(using: .xArbitraryZVertical, to: queue) { [weak self] dm, _ in
            guard let self, let dm else { return }
            let g = dm.gravity
            let gm = max(1e-6, (g.x * g.x + g.y * g.y + g.z * g.z).squareRoot())
            let nx = g.x / gm, ny = g.y / gm, nz = g.z / gm
            let pitch = atan2(-ny, -nz) * 180 / .pi
            let roll  = atan2(-nx, -nz) * 180 / .pi
            let tilt  = acos(min(1, max(-1, -nz))) * 180 / .pi
            let ap = dm.attitude.pitch * 180 / .pi
            let ar = dm.attitude.roll  * 180 / .pi

            let dt = dm.timestamp - self.lastStamp
            self.lastStamp = dm.timestamp
            let inst = (dt > 0 && dt < 1) ? 1.0 / dt : 0

            // Everything below runs on main: the rolling-σ buffers, the
            // capture buffers, and `capturing`/`capStart` are ALL mutated
            // here and nowhere else. (They used to be appended on this
            // background queue while `finish()` read them on main -- an
            // Array data race that could corrupt or crash mid-capture.)
            DispatchQueue.main.async {
                self.pitchDeg = pitch; self.rollDeg = roll; self.tiltDeg = tilt
                self.attPitchDeg = ap; self.attRollDeg = ar

                self.pBuf.append(pitch); if self.pBuf.count > self.bufCap { self.pBuf.removeFirst() }
                self.rBuf.append(roll);  if self.rBuf.count > self.bufCap { self.rBuf.removeFirst() }
                self.pitchSigma = Self.stdDev(self.pBuf)
                self.rollSigma  = Self.stdDev(self.rBuf)
                let s = max(self.pitchSigma, self.rollSigma)
                if s < 0.04 { self.canCapture = true } else if s > 0.12 { self.canCapture = false }
                if inst > 0 { self.rateHz = self.rateHz == 0 ? inst : self.rateHz * 0.9 + inst * 0.1 }

                guard self.capturing else { return }
                if self.waitingForSteady {
                    if self.steady {
                        self.waitingForSteady = false
                        self.capStart = Date()
                        self.capP.removeAll(keepingCapacity: true)
                        self.capR.removeAll(keepingCapacity: true)
                        self.note("wait→accum")
                    } else {
                        // Don't burn the timeout while the phone is being
                        // actively handled -- only a surface that truly
                        // can't settle should give up.
                        if self.pitchSigma > 0.5 || self.rollSigma > 0.5 { self.capArmedAt = Date() }
                        if Date().timeIntervalSince(self.capArmedAt) > self.armTimeout {
                            self.capturing = false
                            self.waitingForSteady = false
                            self.captureRemaining = 0
                            self.armTimedOut = true
                            self.note("TIMEOUT abort")
                        }
                    }
                    return
                }
                self.capP.append(pitch); self.capR.append(roll)
                let el = Date().timeIntervalSince(self.capStart)
                self.captureRemaining = max(0, self.captureSeconds - el)
                if el >= self.captureSeconds { self.finish() }
            }
        }
    }

    func stop() { mm.stopDeviceMotionUpdates() }

    /// Take the next reading in the rosette (or start a fresh one after `.done`).
    func captureRosette() {
        if rosStep == .done { rosReset() }
        begin(seconds: 3)
    }
    /// Clears the whole rosette AND aborts any capture in progress -- the
    /// user's escape hatch from a bad/stuck reading.
    func rosReset() {
        capturing = false; waitingForSteady = false; captureRemaining = 0
        rosStep = .idle; rosCaptures.removeAll(); rosResult = nil; armTimedOut = false
        note("rosReset")
    }

    /// Drop just the last capture and step back one (redo one heading
    /// without redoing the whole rosette).
    func rosUndoLast() {
        guard !capturing, !rosCaptures.isEmpty, rosStep != .done else { return }
        rosCaptures.removeLast()
        rosStep = [.idle, .have1, .have2, .have3][rosCaptures.count]
        armTimedOut = false
        note("rosUndoLast → \(rosCaptures.count)")
    }

    private func begin(seconds: Double) {
        note("begin step=\(rosStep) n=\(rosCaptures.count) cap=\(capturing) stdy=\(steady)")
        guard !capturing else { note("  ↳ blocked, already capturing"); return }
        captureSeconds = seconds
        capP.removeAll(keepingCapacity: true); capR.removeAll(keepingCapacity: true)
        capStart = Date(); capArmedAt = Date(); captureRemaining = seconds
        armTimedOut = false
        // The button is only tappable while `steady`, so normally we drop
        // straight into the accumulation. `waitingForSteady` only covers
        // the case where the tap itself nudged σ over the line.
        waitingForSteady = !steady
        capturing = true
    }

    private func finish() {
        guard capturing else { return }
        capturing = false; waitingForSteady = false; captureRemaining = 0
        let c = Capture(pitch: Self.mean(capP), pitchSD: Self.stdDev(capP),
                        roll: Self.mean(capR), rollSD: Self.stdDev(capR),
                        seconds: Date().timeIntervalSince(capStart), n: capP.count)
        rosCaptures.append(c)
        note("finish n=\(c.n) → rosette \(rosCaptures.count)/4")
        switch rosCaptures.count {
        case 1: rosStep = .have1
        case 2: rosStep = .have2
        case 3: rosStep = .have3
        default:
            rosStep = .done
            let res = RosetteResult.solve(rosCaptures)
            rosResult = res
            readingQueue.append(.rosette(res))
            note(String(format: "  ↳ solved: slope %.2f° res %.3f°", res.slopeMagnitude, res.residualRMS))
        }
    }

    static func mean(_ xs: [Double]) -> Double { xs.isEmpty ? 0 : xs.reduce(0, +) / Double(xs.count) }
    static func stdDev(_ xs: [Double]) -> Double {
        guard xs.count > 1 else { return 0 }
        let m = mean(xs)
        return (xs.reduce(0) { $0 + ($1 - m) * ($1 - m) } / Double(xs.count - 1)).squareRoot()
    }
}

// MARK: - Session model + store

enum Reading: Codable {
    case single(Single)     // legacy — decode only, for older session files
    case flip(Flip)         // legacy — decode only
    case rosette(Rosette)

    var timestamp: Date {
        switch self {
        case .single(let s): return s.timestamp
        case .flip(let f):   return f.timestamp
        case .rosette(let r): return r.timestamp
        }
    }
    var label: String {
        switch self {
        case .single(let s): return s.label
        case .flip(let f):   return f.label
        case .rosette(let r): return r.label
        }
    }

    struct Single: Codable {
        var timestamp: Date
        var label: String
        var pitchDeg, pitchSD, rollDeg, rollSD, seconds: Double
        var samples: Int
    }
    struct Flip: Codable {
        var timestamp: Date
        var label: String
        var truePitchDeg, trueRollDeg, truePitchSD, trueRollSD: Double
        var biasPitchDeg, biasRollDeg: Double
        var aPitchDeg, aRollDeg, aPitchSD, aRollSD: Double
        var bPitchDeg, bRollDeg, bPitchSD, bRollSD: Double
        var aSamples, bSamples: Int
        var suspectFlippedOver: Bool
    }
    struct Rosette: Codable {
        var timestamp: Date
        var label: String
        var slopeMagnitudeDeg: Double
        var downhillDirDeg: Double            // 0 deg = capture-1 top edge, clockwise+
        var truePitchDeg, trueRollDeg, truePitchSD, trueRollSD: Double  // capture-1 frame
        var biasPitchDeg, biasRollDeg: Double
        var residualRMSDeg: Double
        var suspect: Bool
        var pitchDeg, rollDeg, pitchSD, rollSD: [Double]   // 4 raw captures, capture order
        var samples: [Int]
    }

    private enum K: String, CodingKey { case kind }
    init(from d: Decoder) throws {
        let c = try d.container(keyedBy: K.self)
        switch try c.decode(String.self, forKey: .kind) {
        case "rosette": self = .rosette(try Rosette(from: d))
        case "flip":    self = .flip(try Flip(from: d))
        default:        self = .single(try Single(from: d))
        }
    }
    func encode(to e: Encoder) throws {
        var c = e.container(keyedBy: K.self)
        switch self {
        case .single(let s):  try c.encode("single", forKey: .kind);  try s.encode(to: e)
        case .flip(let f):    try c.encode("flip", forKey: .kind);    try f.encode(to: e)
        case .rosette(let r): try c.encode("rosette", forKey: .kind); try r.encode(to: e)
        }
    }
}

struct Session: Codable {
    var name: String
    var createdAt: Date
    var device: String
    var appVersion: String
    var readings: [Reading]
}

final class SessionStore: ObservableObject {
    struct File: Identifiable {
        let url: URL
        var session: Session
        // Identity is the FILENAME, not the URL. URLs from
        // contentsOfDirectory(at:) don't reliably compare `==` to a URL
        // built with appendingPathComponent (symlink/encoding differences),
        // which was silently breaking "is this the active session?".
        var key: String { url.lastPathComponent }
        var id: String { key }
    }
    @Published private(set) var files: [File] = []

    let directory: URL = {
        let d = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Sessions", isDirectory: true)
        try? FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        return d
    }()
    private let enc: JSONEncoder = {
        let e = JSONEncoder(); e.dateEncodingStrategy = .iso8601
        e.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]; return e
    }()
    private let dec: JSONDecoder = {
        let d = JSONDecoder(); d.dateDecodingStrategy = .iso8601; return d
    }()

    init() { reload() }

    func reload() {
        let urls = (try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)) ?? []
        files = urls.filter { $0.pathExtension == "json" }
            .compactMap { u -> File? in
                guard let data = try? Data(contentsOf: u),
                      let s = try? dec.decode(Session.self, from: data) else { return nil }
                return File(url: u, session: s)
            }
            .sorted { $0.session.createdAt > $1.session.createdAt }
    }

    /// Creates the file and returns its key (filename).
    @discardableResult
    func create(name: String) -> String {
        let s = Session(name: name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "Session" : name,
                        createdAt: Date(), device: Self.deviceModel(),
                        appVersion: Self.appVersion(), readings: [])
        let key = Self.filename(for: s)
        try? enc.encode(s).write(to: directory.appendingPathComponent(key), options: .atomic)
        reload()
        return key
    }

    func file(key: String) -> File? { files.first { $0.key == key } }
    func session(key: String) -> Session? { file(key: key)?.session }

    func append(_ r: Reading, key: String) {
        // Re-read the file from disk, don't trust the in-memory copy -- an
        // append that lands before a prior reload finished would otherwise
        // start from a stale reading list and silently drop readings.
        let url = directory.appendingPathComponent(key)
        guard let data = try? Data(contentsOf: url),
              var s = try? dec.decode(Session.self, from: data) else { return }
        s.readings.append(r)
        try? enc.encode(s).write(to: url, options: .atomic)
        reload()
    }

    func delete(key: String) {
        if let f = file(key: key) { try? FileManager.default.removeItem(at: f.url) }
        reload()
    }

    /// Remove one reading from a session (re-reads disk first, same as append).
    func deleteReading(at index: Int, key: String) {
        let url = directory.appendingPathComponent(key)
        guard let data = try? Data(contentsOf: url),
              var s = try? dec.decode(Session.self, from: data),
              s.readings.indices.contains(index) else { return }
        s.readings.remove(at: index)
        try? enc.encode(s).write(to: url, options: .atomic)
        reload()
    }

    static func filename(for s: Session) -> String {
        let df = DateFormatter(); df.dateFormat = "yyyyMMdd-HHmmss"
        let safe = s.name.replacingOccurrences(of: "/", with: "-")
            .replacingOccurrences(of: ":", with: "-")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let base = safe.isEmpty ? "session" : String(safe.prefix(40))
        return "\(base)_\(df.string(from: s.createdAt)).json"
    }
    static func deviceModel() -> String {
        var s = utsname(); uname(&s)
        return withUnsafePointer(to: &s.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: 1) { String(cString: $0) }
        }
    }
    static func appVersion() -> String {
        (Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String) ?? "1.0"
    }
}

// MARK: - UI

struct ContentView: View {
    @StateObject private var inc = Inclinometer()
    @StateObject private var store = SessionStore()

    // The active session's key = its filename. Persisted, so it survives an
    // app relaunch. Empty string = no active session.
    @AppStorage("activeSessionFile") private var activeKey = ""
    @State private var readingLabel = ""

    @State private var showSessions = false
    @State private var showNew = false
    @State private var newName = ""

    private var activeSession: Session? { activeKey.isEmpty ? nil : store.session(key: activeKey) }

    var body: some View {
        ScrollView {
            VStack(spacing: 12) {
                HStack {
                    Text("GROUND LEVEL").font(.caption).kerning(3).foregroundStyle(.secondary)
                    Spacer()
                    Button { showSessions = true } label: {
                        Label("Sessions", systemImage: "folder").font(.caption)
                    }
                }
                .padding(.horizontal, 22)

                sessionBar

                HStack(spacing: 8) {
                    Circle().fill(inc.steady ? .green : .orange).frame(width: 9, height: 9)
                    Text(inc.steady ? "steady — flat, screen up"
                         : "settling… hold still").font(.caption).foregroundStyle(.secondary)
                }

                reading("PITCH", inc.pitchDeg, note: "fore / aft   ·   + = top edge up")
                reading("ROLL",  inc.rollDeg,  note: "side   ·   + = right edge up")

                Text(String(format: "tilt %.2f° from level    σₚ ±%.3f°   σᵣ ±%.3f°    %.0f Hz",
                            inc.tiltDeg, inc.pitchSigma, inc.rollSigma, inc.rateHz))
                    .font(.system(.footnote, design: .monospaced)).foregroundStyle(.secondary)
                Text(String(format: "CMAttitude  p %+.2f°   r %+.2f°   (≈ DriverAssist Level)",
                            inc.attPitchDeg, inc.attRollDeg))
                    .font(.system(.caption2, design: .monospaced)).foregroundStyle(.tertiary)

                if activeSession != nil {
                    TextField("reading label (optional) — e.g. Front left", text: $readingLabel)
                        .textFieldStyle(.roundedBorder)
                        .autocorrectionDisabled()
                        .padding(.horizontal, 22)
                }

                rosetteSection

                debugPanel
            }
            .padding(.top, 8)
            .padding(.bottom, 24)
            .frame(maxWidth: .infinity)
        }
        .background(Color(.systemBackground))
        .onAppear {
            inc.start()
            store.reload()
            if !activeKey.isEmpty && store.file(key: activeKey) == nil { activeKey = "" }
        }
        .onDisappear { inc.stop() }
        .onChange(of: inc.readingQueue) { drainReadings() }
        .sheet(isPresented: $showSessions) {
            SessionsView(store: store, activeKey: $activeKey) { key in
                activeKey = key
                inc.clearPending()
                if !key.isEmpty { showSessions = false }
            }
        }
        .alert("New session", isPresented: $showNew) {
            TextField("Name", text: $newName)
            Button("Create") { activeKey = store.create(name: newName); inc.clearPending(); newName = "" }
            Button("Cancel", role: .cancel) { newName = "" }
        } message: {
            Text("Every rosette is appended to this session's JSON file.")
        }
    }

    // MARK: debug panel (temporary — verifying the rosette flow)

    private var debugPanel: some View {
        VStack(alignment: .leading, spacing: 1) {
            Text("step=\(String(describing: inc.rosStep))  n=\(inc.rosCaptures.count)  cap=\(inc.capturing ? 1 : 0)  wait=\(inc.waitingForSteady ? 1 : 0)  canCap=\(inc.canCapture ? 1 : 0)")
                .foregroundStyle(.orange)
            ForEach(Array(inc.log.enumerated()), id: \.offset) { _, line in
                Text(line).foregroundStyle(.tertiary)
            }
        }
        .font(.system(size: 10, design: .monospaced))
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(8)
        .background(.quaternary, in: RoundedRectangle(cornerRadius: 8))
        .padding(.horizontal, 22)
    }

    // MARK: session bar

    private var sessionBar: some View {
        HStack(spacing: 10) {
            Image(systemName: "square.and.pencil")
            if let s = activeSession {
                VStack(alignment: .leading, spacing: 1) {
                    Text("Active session: \(s.name)").font(.subheadline).bold().lineLimit(1)
                    Text("\(s.readings.count) reading\(s.readings.count == 1 ? "" : "s") · started \(s.createdAt.formatted(date: .abbreviated, time: .shortened))")
                        .font(.caption2).foregroundStyle(.secondary)
                }
                Spacer()
                Button("End session") { activeKey = "" }.font(.caption)
            } else {
                Text("No active session").font(.subheadline).foregroundStyle(.secondary)
                Spacer()
                Button("New") { showNew = true }.font(.caption).bold()
            }
        }
        .padding(10)
        .background(.quaternary, in: RoundedRectangle(cornerRadius: 12))
        .padding(.horizontal, 22)
    }

    // MARK: rosette capture

    private var rosStepIndex: Int {
        switch inc.rosStep {
        case .idle:  return 0
        case .have1: return 1
        case .have2: return 2
        case .have3: return 3
        case .done:  return 4
        }
    }
    private var rosetteHint: String {
        switch inc.rosStep {
        case .idle:  return "Capture 1 of 4 — phone flat, top edge toward the front of the car."
        case .have1: return "Rotate the phone 90° CLOCKWISE (flat, same spot). Capture 2 of 4."
        case .have2: return "Rotate 90° CLOCKWISE again. Capture 3 of 4."
        case .have3: return "Rotate 90° CLOCKWISE again — last one. Capture 4 of 4."
        case .done:  return "Saved. Capture 1 again for the next spot, or Reset."
        }
    }
    private var rosetteCTA: String {
        switch inc.rosStep {
        case .idle, .done: return "Capture 1 — 3 s"
        case .have1:       return "Capture 2 — 3 s"
        case .have2:       return "Capture 3 — 3 s"
        case .have3:       return "Capture 4 — 3 s"
        }
    }

    private var rosetteSection: some View {
        VStack(spacing: 12) {
            Text(rosetteHint).font(.footnote).foregroundStyle(.secondary).multilineTextAlignment(.center)

            HStack(spacing: 12) {
                ForEach(0..<4) { i in
                    // Cleared once the rosette is done, ready for the next spot.
                    let filled = inc.rosStep != .done && i < rosStepIndex
                    Image(systemName: filled ? "\(i + 1).circle.fill" : "\(i + 1).circle")
                        .foregroundStyle(filled ? Color.accentColor : Color.secondary)
                }
            }
            .font(.title2)
            .imageScale(.large)

            captureButton(rosetteCTA) { inc.captureRosette() }

            if inc.rosStep != .idle || inc.capturing {
                HStack(spacing: 20) {
                    if inc.rosCaptures.count > 0 && inc.rosCaptures.count < 4 && !inc.capturing {
                        Button("Redo last") { inc.rosUndoLast() }
                    }
                    Button("Reset — start this spot over", role: .destructive) { inc.rosReset() }
                }
                .font(.footnote)
            }

            if let r = inc.rosResult, inc.rosStep == .done {
                card {
                    Text("ROSETTE FIT").font(.caption2).kerning(3).foregroundStyle(.secondary)
                    Text(String(format: "slope   %.2f°     downhill  %.0f° CW", r.slopeMagnitude, r.downhillDirDeg))
                    Text(String(format: "pitch  %+.3f°  ± %.3f°", r.truePitch, r.truePitchSD))
                    Text(String(format: "roll   %+.3f°  ± %.3f°", r.trueRoll, r.trueRollSD))
                    Divider().padding(.vertical, 2)
                    Text(String(format: "bias   p %+.3f°   r %+.3f°", r.biasPitch, r.biasRoll))
                        .font(.system(.footnote, design: .monospaced)).foregroundStyle(.secondary)
                    Text(String(format: "residual  %.3f°   (turn quality)", r.residualRMS))
                        .font(.system(.caption2, design: .monospaced)).foregroundStyle(.tertiary)
                    ForEach(0..<r.caps.count, id: \.self) { i in
                        Text(String(format: "%d.  p %+.3f   r %+.3f", i + 1, r.caps[i].pitch, r.caps[i].roll))
                            .font(.system(.caption2, design: .monospaced)).foregroundStyle(.tertiary)
                    }
                    if r.suspect {
                        Text("⚠︎ residual is high — a 90° turn was probably off, or the spot isn't planar. Re-take.")
                            .font(.caption2).foregroundStyle(.orange).multilineTextAlignment(.center)
                    }
                }
            }
        }
        .padding(.horizontal, 22)
    }

    // MARK: append — drains every queued reading, so none is lost if two land at once

    private func drainReadings() {
        guard !inc.readingQueue.isEmpty else { return }
        let batch = inc.readingQueue
        inc.readingQueue.removeAll()
        guard !activeKey.isEmpty else { inc.note("drain: no active session, dropped \(batch.count)"); return }
        for payload in batch { appendPayload(payload) }
        inc.note("drain: appended \(batch.count) → \(activeKey.prefix(12))")
        readingLabel = ""
    }

    private func appendPayload(_ payload: Inclinometer.ReadingPayload) {
        let now = Date()
        switch payload {
        case .rosette(let r):
            store.append(.rosette(.init(
                timestamp: now, label: readingLabel,
                slopeMagnitudeDeg: r.slopeMagnitude, downhillDirDeg: r.downhillDirDeg,
                truePitchDeg: r.truePitch, trueRollDeg: r.trueRoll,
                truePitchSD: r.truePitchSD, trueRollSD: r.trueRollSD,
                biasPitchDeg: r.biasPitch, biasRollDeg: r.biasRoll,
                residualRMSDeg: r.residualRMS, suspect: r.suspect,
                pitchDeg: r.caps.map { $0.pitch }, rollDeg: r.caps.map { $0.roll },
                pitchSD:  r.caps.map { $0.pitchSD }, rollSD: r.caps.map { $0.rollSD },
                samples:  r.caps.map { $0.n })), key: activeKey)
        }
    }

    // MARK: bits

    private func reading(_ label: String, _ value: Double, note: String) -> some View {
        VStack(spacing: 2) {
            HStack(alignment: .firstTextBaseline, spacing: 14) {
                Text(label).font(.system(.headline, design: .rounded)).foregroundStyle(.secondary)
                    .frame(width: 66, alignment: .leading)
                Text(String(format: "%+.2f°", value))
                    .font(.system(size: 50, weight: .semibold, design: .monospaced))
                    .monospacedDigit().contentTransition(.numericText())
                    .animation(.default, value: value)
            }
            Text(note).font(.caption2).foregroundStyle(.tertiary)
        }
    }
    private func captureButton(_ title: String, _ action: @escaping () -> Void) -> some View {
        VStack(spacing: 4) {
            Button(action: action) {
                Text(buttonLabel(title))
                    .font(.headline).frame(maxWidth: .infinity).padding(.vertical, 12)
            }
            .buttonStyle(.borderedProminent)
            .disabled(inc.capturing || !inc.canCapture)   // must be steady to start (hysteretic)
            if inc.armTimedOut {
                Text("couldn't get a steady reading — try again")
                    .font(.caption2).foregroundStyle(.orange)
            }
        }
    }
    private func buttonLabel(_ title: String) -> String {
        if inc.capturing {
            return inc.waitingForSteady ? "hold still…"
                : String(format: "capturing…  %.1f s", inc.captureRemaining)
        }
        return inc.canCapture ? title : "hold still to capture"
    }
    private func card<Content: View>(@ViewBuilder _ content: () -> Content) -> some View {
        VStack(spacing: 6, content: content)
            .font(.system(.title3, design: .monospaced))
            .textSelection(.enabled)
            .padding(16).frame(maxWidth: .infinity)
            .background(.quaternary, in: RoundedRectangle(cornerRadius: 14))
    }
}

// MARK: - Sessions browser

struct SessionsView: View {
    @ObservedObject var store: SessionStore
    @Binding var activeKey: String
    var onPick: (String) -> Void            // "" = clear active

    @State private var showNew = false
    @State private var newName = ""
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                if store.files.isEmpty {
                    Text("No saved sessions yet.").foregroundStyle(.secondary)
                }
                ForEach(store.files) { f in
                    NavigationLink {
                        SessionDetailView(store: store, fileKey: f.key, isActive: activeKey == f.key) { onPick(f.key) }
                    } label: {
                        VStack(alignment: .leading, spacing: 2) {
                            HStack {
                                Text(f.session.name).bold()
                                if activeKey == f.key {
                                    Text("active").font(.caption2).padding(.horizontal, 6).padding(.vertical, 1)
                                        .background(.green.opacity(0.25), in: Capsule())
                                }
                            }
                            Text("\(f.session.readings.count) reading\(f.session.readings.count == 1 ? "" : "s") · \(f.session.createdAt.formatted(date: .abbreviated, time: .shortened))")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                    }
                }
                .onDelete { idx in
                    for i in idx {
                        let key = store.files[i].key
                        if activeKey == key { onPick("") }
                        store.delete(key: key)
                    }
                }
            }
            .navigationTitle("Sessions")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) { Button("Done") { dismiss() } }
                ToolbarItem(placement: .topBarTrailing) { Button { showNew = true } label: { Image(systemName: "plus") } }
            }
            .alert("New session", isPresented: $showNew) {
                TextField("Name", text: $newName)
                Button("Create") { onPick(store.create(name: newName)); newName = "" }
                Button("Cancel", role: .cancel) { newName = "" }
            }
        }
    }
}

struct SessionDetailView: View {
    @ObservedObject var store: SessionStore
    let fileKey: String
    let isActive: Bool
    var makeActive: () -> Void
    @Environment(\.dismiss) private var dismiss

    private var file: SessionStore.File? { store.file(key: fileKey) }

    var body: some View {
        Group {
            if let file {
                List {
                    Section {
                        LabeledContent("Created", value: file.session.createdAt.formatted(date: .long, time: .standard))
                        LabeledContent("Device", value: file.session.device)
                        LabeledContent("Readings", value: "\(file.session.readings.count)")
                        LabeledContent("File", value: file.url.lastPathComponent)
                    }
                    Section("Readings") {
                        ForEach(Array(file.session.readings.enumerated()), id: \.offset) { _, r in
                            readingRow(r)
                        }
                        .onDelete { offsets in
                            for i in offsets.sorted(by: >) { store.deleteReading(at: i, key: fileKey) }
                        }
                        if file.session.readings.isEmpty { Text("None yet.").foregroundStyle(.secondary) }
                    }
                }
                .navigationTitle(file.session.name)
                .navigationBarTitleDisplayMode(.inline)
                .toolbar {
                    ToolbarItem(placement: .topBarTrailing) { ShareLink(item: file.url) }
                    ToolbarItem(placement: .topBarTrailing) {
                        if !file.session.readings.isEmpty { EditButton() }
                    }
                    ToolbarItem(placement: .bottomBar) {
                        Button(isActive ? "Currently active" : "Make active session") { makeActive(); dismiss() }
                            .disabled(isActive)
                    }
                }
            } else {
                ContentUnavailableView("Session deleted", systemImage: "trash")
            }
        }
    }

    @ViewBuilder private func readingRow(_ r: Reading) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack {
                Text(r.timestamp.formatted(date: .omitted, time: .standard)).font(.caption).foregroundStyle(.secondary)
                if !r.label.isEmpty { Text(r.label).font(.caption).bold() }
                Spacer()
                switch r {
                case .single:  Text("single").font(.caption2).foregroundStyle(.tertiary)
                case .flip:    Text("flip").font(.caption2).foregroundStyle(.tertiary)
                case .rosette: Text("rosette").font(.caption2).foregroundStyle(.tertiary)
                }
            }
            switch r {
            case .single(let s):
                Text(String(format: "pitch %+.3f° ±%.3f    roll %+.3f° ±%.3f", s.pitchDeg, s.pitchSD, s.rollDeg, s.rollSD))
                    .font(.system(.footnote, design: .monospaced))
            case .flip(let f):
                Text(String(format: "TRUE pitch %+.3f° ±%.3f    roll %+.3f° ±%.3f", f.truePitchDeg, f.truePitchSD, f.trueRollDeg, f.trueRollSD))
                    .font(.system(.footnote, design: .monospaced))
                Text(String(format: "bias  p %+.3f  r %+.3f", f.biasPitchDeg, f.biasRollDeg))
                    .font(.system(.caption2, design: .monospaced)).foregroundStyle(.secondary)
            case .rosette(let r):
                Text(String(format: "slope %.2f°  downhill %.0f° CW    pitch %+.3f° ±%.3f  roll %+.3f° ±%.3f",
                            r.slopeMagnitudeDeg, r.downhillDirDeg, r.truePitchDeg, r.truePitchSD, r.trueRollDeg, r.trueRollSD))
                    .font(.system(.footnote, design: .monospaced))
                Text(String(format: "bias  p %+.3f  r %+.3f    residual %.3f°%@",
                            r.biasPitchDeg, r.biasRollDeg, r.residualRMSDeg, r.suspect ? "   ⚠︎" : ""))
                    .font(.system(.caption2, design: .monospaced)).foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 2)
    }
}
