# GroundLevel

Tiny iOS app: put the phone flat on a spot and read how far that spot is
from level, at the iPhone inclinometer's own precision.

Built to check whether the `26_09_01_ConeAndCar` calibration spot is
sloped — a ~1–2° ground tilt relative to the plane the camera
height/pitch calibration assumes would explain a row-based distance error
that recalibrating height/pitch can't fix.

## Build / install

Open `GroundLevel.xcodeproj` and run to your device, or from the CLI:

```bash
cd calibration/GroundLevelApp
xcodebuild -project GroundLevel.xcodeproj -scheme GroundLevel -configuration Debug \
  -destination 'platform=iOS,id=<DEVICE-UDID>' -derivedDataPath build \
  -allowProvisioningUpdates build
xcrun devicectl device install app --device <DEVICE-UDID> \
  build/Build/Products/Debug-iphoneos/GroundLevel.app
```

Signed with team `J969NUK675`. No capabilities / usage strings needed —
`CMDeviceMotion` requires no permission.

## Rosette (4×90°) — the only capture mode

One reading = **four 3-second captures at the same spot**, rotating the
phone **90° clockwise** (viewed from above) between each:

1. Phone flat, screen up, top edge toward the front of the car. **Capture 1**.
2. Rotate 90° **clockwise**, flat, same spot. **Capture 2**.
3. Rotate 90° clockwise again. **Capture 3**.
4. Rotate 90° clockwise again. **Capture 4** → the fit runs.

The capture button is disabled until the reading is **steady**; after each
turn, wait ~1–2 s for it to settle (it reads "hold still to capture").

### What the fit does

With the phone rotated clockwise by `i·90°` and a fixed phone-frame bias
`(Bp, Br)` plus a ground-slope vector `(gx, gy)` in capture 1's frame:

```
p_i = Bp + gx·cos(i·90°) + gy·sin(i·90°)
r_i = Br − gx·sin(i·90°) + gy·cos(i·90°)
```

Least squares over the 8 numbers (4 pitch + 4 roll) → `Bp, Br, gx, gy`.
For 4 evenly-spaced headings this is the closed form

```
Bp = mean(pᵢ)                       Br = mean(rᵢ)
gx = ((p₀−p₂) + (r₃−r₁)) / 4         gy = ((p₁−p₃) + (r₀−r₂)) / 4
```

- **true pitch / roll** = `gx` / `gy` — the ground slope in capture 1's
  orientation, phone bias removed.
- **slope magnitude** = `hypot(gx, gy)`.
- **downhill direction** = `atan2(−gy, −gx)`, in degrees **clockwise from
  capture 1's top edge** (0° = top edge, 90° = right edge, 180° = bottom,
  270° = left).
- **residual RMS** = RMS of the 8 fit residuals. Sensor noise alone is
  ~0.005°, so a clean rosette lands well under ~0.05°. A high residual
  (flagged ⚠︎ above ~0.15°) means a 90° turn was off, the phone tilted
  differently in one heading, or the spot isn't planar — re-take it.

Four headings beat a single 180° flip: per-capture noise averages down,
small turn-angle errors wash out, and a bad turn shows up in the residual
instead of silently biasing the answer.

> **Sign check on first real use:** if the four individual captures are
> clean (tight σ) but the residual comes out roughly as large as the slope
> itself, the clockwise-rotation sign in the model is inverted for this
> device's motion frame — negate the `sin` terms (one-line fix).

## Sessions

Tap **New** (session bar), name the run. From then on **every rosette is
appended** to that session's JSON file under `Documents/Sessions/`, each
with its own timestamp and optional label (type it in the "reading label"
field before the 4th capture). The file also stores its creation date,
the device model, and the app version.

- **End session** stops appending; the active session survives a relaunch.
- **Sessions** (top-right) lists every saved file: open one to see all
  readings, **share** it (AirDrop / Save to Files / Mail), or swipe to
  delete. **Make active** to resume adding to it.
- Each session is a standalone `.json` — many can be saved.

### File format

```json
{
  "name": "Parking Spot",
  "createdAt": "2026-09-03T02:15:48Z",
  "device": "iPhone18,2",
  "appVersion": "1.0",
  "readings": [
    { "kind": "rosette", "timestamp": "…", "label": "Front left",
      "slopeMagnitudeDeg": 1.83, "downhillDirDeg": 142.0,
      "truePitchDeg": 1.552, "trueRollDeg": -0.821,
      "truePitchSD": 0.003, "trueRollSD": 0.004,
      "biasPitchDeg": 2.01, "biasRollDeg": 0.78,
      "residualRMSDeg": 0.031, "suspect": false,
      "pitchDeg": [ …4 raw… ], "rollDeg": [ …4… ],
      "pitchSD":  [ …4… ],     "rollSD":  [ …4… ],
      "samples":  [ 300, 300, 300, 300 ] }
  ]
}
```

Readings are ordered `capture 1 → 4`. Older session files may also contain
`"kind": "single"` and `"kind": "flip"` readings — the app still reads
them, it just no longer writes them.

Flat keys, ISO8601 timestamps, sorted keys — `pandas.json_normalize` on
`session["readings"]` works (array-valued columns hold the 4 raw captures).

## Reading it

- **σ** on true pitch/roll is *precision* (repeatability), typically
  ~0.003° after the 4-way average. Absolute accuracy of the slope is set
  by how well you hit each 90° turn — the residual is your check.
- `CMAttitude p, r` on the live screen is the same quantity DriverAssist
  logs as `referencePitchDegrees` / `referenceRollDegrees`.
- Signs: `+pitch` = top edge raised, `+roll` = right edge raised.

## App icon

`make_icon.py` renders the spirit-level icon to
`Assets.xcassets/AppIcon.appiconset/icon-1024.png`. Re-run after edits:

```bash
python3 calibration/GroundLevelApp/make_icon.py
```
