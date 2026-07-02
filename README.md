# Image Background Cleaner

A lightweight desktop tool for cleaning simple image backgrounds and exporting PNG files.

The app is built with Python, PySide6, Pillow, and numpy. It supports importing PNG/JPG images, previewing the original and processed result side by side, removing or replacing detected background pixels, and exporting the result as PNG.

## Features

- Import PNG, JPG, and JPEG images
- Drag and drop image import
- Side-by-side original and processed previews
- Mouse-wheel zoom on the processed preview for edge/detail inspection, with right-drag pan and reset
- Drag-select a crop area on the original preview, then refine it from the four corner handles
- Export a cropped image directly without running background removal
- Reset a cropped image back to the imported original
- Automatically detect the main subject by combining multiple mask candidates
- Edit the subject edge by dragging an add/remove brush over the original preview
- Pick a target background color with a color picker
- Remove pixels near the selected color with tolerance control
- Remove light or gray low-saturation backgrounds
- Export transparent PNG files
- Replace the detected background with a selected solid color
- Crop the processed result into 1 inch, 2 inch, or custom-size ID photos
- Preview transparent results on checkerboard, dark, blue, or yellow backgrounds
- One-click Windows launcher with `start.bat`

## Requirements

- Python 3.10 or newer
- PySide6
- Pillow
- numpy
- opencv-python
- onnxruntime CPU backend

Install dependencies:

```powershell
python -m pip install -r requirements.txt
python download_models.py
```

`download_models.py` stores the subject-detection ONNX model under
`models/rembg`. After that setup step, the app does not need network access while
running. If the local model is missing or fails its hash check, subject detection
falls back to the bundled OpenCV-based detector instead of downloading anything
at runtime.
If you package the app as an installer, include
`models/rembg/u2net_human_seg.onnx` in the installed files. The ONNX file is a
local install artifact and is ignored by git to avoid accidental large commits.

## Start The App

On Windows, double-click:

```text
start.bat
```

Or run from PowerShell:

```powershell
python main.py
```

The app opens maximized by default.

## Basic Workflow

1. Click `Drop image here or click`, or drag an image into the app.
2. Optional: click `Select Crop Area`, drag over the original preview, adjust the corner handles if needed, then click `Crop Image`.
3. Optional: click `Detect Subject` to automatically find the main subject. Use `Edit Subject Edge` with the add/remove brush if the edge needs correction, then click `Apply Subject`.
4. Choose a background removal mode:
   - `Picked color`: removes pixels near the target color.
   - `Light/gray background`: removes bright, low-saturation background pixels.
5. Click the `Target color` swatch to choose the background color to detect.
6. Adjust `Color tolerance` or `Light cutoff` if needed.
7. Choose an output mode:
   - `Transparent`: exports with alpha transparency.
   - `Solid color`: replaces the detected background with the selected background color.
8. Click `Apply Background`. If a subject mask is active, the app uses that mask to make the background transparent or replace it with the selected solid color. If no subject mask is active, it uses the selected color/light background mode.
9. Optional: in `ID Photo`, choose `1 inch`, `2 inch`, or `Custom`, then click `Select ID Photo Crop`. Move or resize the fixed crop box on the processed preview, or mouse-wheel zoom and right-drag the image underneath it, then click `Apply ID Photo Crop`.
10. Click `Export PNG`.

Cropping keeps the selected rectangle as-is. It does not stretch the image or force the crop into a square.

## Notes

Subject detection first uses the locally cached human-segmentation model when it
is installed, then falls back to deterministic image-processing rules:

- color-distance matching for colored backgrounds
- brightness and low-saturation matching for light or gray backgrounds
- combined OpenCV GrabCut, border-background contrast, and center-prior subject masks for complex backgrounds, with manual edge editing

It works best for stamps, signatures, line art, green/blue screen images, and images where the subject is visually distinct from the background.

## Project Structure

```text
main.py              Application entry point
image_processor.py   Pillow/numpy image processing logic
download_models.py   Installation-time model preloader for offline detection
ui/main_window.py    PySide6 main window and interactions
ui/theme.py          Application styling
start.bat            Windows one-click launcher
requirements.txt     Python dependencies
```
