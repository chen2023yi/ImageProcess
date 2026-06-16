# Image Background Cleaner

A lightweight desktop tool for cleaning simple image backgrounds and exporting PNG files.

The app is built with Python, PySide6, Pillow, and numpy. It supports importing PNG/JPG images, previewing the original and processed result side by side, removing or replacing detected background pixels, and exporting the result as PNG.

## Features

- Import PNG, JPG, and JPEG images
- Drag and drop image import
- Side-by-side original and processed previews
- Pick a target background color with a color picker
- Remove pixels near the selected color with tolerance control
- Remove light or gray low-saturation backgrounds
- Export transparent PNG files
- Replace the detected background with a selected solid color
- Preview transparent results on checkerboard, dark, blue, or yellow backgrounds
- One-click Windows launcher with `start.bat`

## Requirements

- Python 3.10 or newer
- PySide6
- Pillow
- numpy

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

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
2. Choose a background removal mode:
   - `Picked color`: removes pixels near the target color.
   - `Light/gray background`: removes bright, low-saturation background pixels.
3. Click the `Target color` swatch to choose the background color to detect.
4. Adjust `Color tolerance` or `Light cutoff` if needed.
5. Choose an output mode:
   - `Transparent`: exports with alpha transparency.
   - `Solid color`: replaces the detected background with the selected background color.
6. Click `Apply Background`.
7. Click `Export PNG`.

## Notes

This is not an AI background remover. It uses deterministic image-processing rules:

- color-distance matching for colored backgrounds
- brightness and low-saturation matching for light or gray backgrounds

It works best for stamps, signatures, line art, green/blue screen images, and images where the subject is visually distinct from the background.

## Project Structure

```text
main.py              Application entry point
image_processor.py   Pillow/numpy image processing logic
ui/main_window.py    PySide6 main window and interactions
ui/theme.py          Application styling
start.bat            Windows one-click launcher
requirements.txt     Python dependencies
```
