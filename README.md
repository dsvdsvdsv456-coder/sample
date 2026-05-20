# Touchless AI Drawing Studio

Touchless AI Drawing Studio is a webcam-based air drawing, whiteboard, notes,
and diagram app built with OpenCV and the modern MediaPipe Tasks Hand
Landmarker API.

This project intentionally does not use `mp.solutions` or
`mp.solutions.hands`.

## Setup

Run these commands in PowerShell from the project folder:

```powershell
pip install -r requirements.txt
python download_model.py
python main.py
```

## Phase 12: AI Diagram Assistant

Diagram Mode converts rough whiteboard sketches into clean structured diagrams.

Press `D` to toggle:

```text
Diagram Mode: ON/OFF
```

When Diagram Mode is ON, rough strokes are analyzed as diagram parts:

- Rectangle = `PROCESS`
- Circle = `START/END`
- Line = `CONNECTION`

The app stores detected diagram objects separately from normal pen drawing. The
UI shows:

```text
Diagram Mode ON/OFF
Objects detected: count
```

## How Diagram Detection Works

The app already captures one completed stroke after `DRAW_LOCK` ends. In Diagram
Mode, that stroke goes through `ShapeDetector`:

- `RECTANGLE` becomes a process box.
- `CIRCLE` becomes a start/end node.
- `LINE` becomes a possible connection.

Those results are stored by `utils/diagram_engine.py`.

## How Connections Work

When a line is detected, the diagram engine checks which two shapes are closest
to the line endpoints. If the line starts near one shape and ends near another,
it becomes an arrow connection.

If you draw multiple shapes but no usable line, the engine still connects
neighboring shapes during layout so the demo remains useful.

## Auto Layout

Press `ENTER` while Diagram Mode is ON.

The engine:

1. Restores the canvas to how it looked before rough diagram strokes.
2. Aligns detected shapes in a clean row.
3. Applies equal spacing.
4. Centers them vertically.
5. Redraws perfect rectangles/circles.
6. Draws straight arrows with arrowheads.

## Core Tools

- `PEN`: freehand drawing.
- `ERASER`: remove strokes.
- `SHAPE`: Smart Shape correction mode.
- `TEXT`: place typed text or quick labels.
- `DIAGRAM MODE`: detect shapes and build a diagram model.

Normal drawing is unchanged when Diagram Mode is OFF.

## Keyboard Shortcuts

```text
D = Diagram Mode ON/OFF
ENTER = auto-layout diagram when Diagram Mode is ON
P = pen
E = eraser
A = shape
L = text
6-9 = quick note labels
W = Camera / Whiteboard mode
B = White / Dark board
N = new page
Left Arrow = previous page
Right Arrow = next page
Z or Ctrl+Z = undo
Y or Ctrl+Y = redo
T = Smart Shape ON/OFF
C = clear current page
S = save current page canvas
Shift+S or X = screenshot
1-5 = colors
M = camera enhancement ON/OFF
V = visual mode
H = landmarks ON/OFF
Q = quit
```

## How To Test Phase 12

1. Run:

```powershell
python main.py
```

2. Press `W` for Whiteboard mode.
3. Press `D` to turn Diagram Mode ON.
4. Draw a rough rectangle.
5. Draw a second rough rectangle.
6. Draw a line between them.
7. Press `ENTER`.
8. Expect a clean aligned diagram with two process boxes and a straight arrow.

Try a flowchart:

- Circle
- Rectangle
- Rectangle
- Line between each
- Press `ENTER`

The result should become a clean, evenly spaced diagram.

## Common Problems

- Nothing happens on `ENTER`:
  Make sure Diagram Mode is ON and at least one object was detected.

- A rough shape is not detected:
  Draw a larger, clearer rectangle/circle/line.

- Connection is missing:
  Start and end the line closer to the shapes you want to connect.

- Normal drawing changed:
  Press `D` to turn Diagram Mode OFF. Normal PEN/ERASER/TEXT behavior is
  unchanged outside Diagram Mode.

## Project Structure

```text
main.py
download_model.py
requirements.txt
models/
utils/
  diagram_engine.py
  hand_tracker.py
  gesture_detector.py
  drawing_canvas.py
  page_manager.py
  shape_detector.py
  text_tool.py
  ui_toolbar.py
  smoothing.py
  visual_overlay.py
assets/
  saved_drawings/
```

## Model File

The hand model is downloaded to:

```text
models/hand_landmarker.task
```

If the model is missing, run:

```powershell
python download_model.py
```
