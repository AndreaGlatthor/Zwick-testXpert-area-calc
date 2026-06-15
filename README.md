# Zwick testXpert Area Calculator

This project is a web application that evaluates a T-peel test curve from a Zwick testXpert screenshot. 
It is useful if you do not have the option of integrating with testXpert or exporting to Excel, 
both of which require a license.

## What the app is for

The app integrates the area under the curve and also calculates the average force.

- Digitizes the blue force-displacement curve from a screenshot.
- Converts pixel coordinates to physical units:
  - X-axis: displacement in mm
  - Y-axis: force in N
- Calculates the integrated area in a selected displacement interval.
- Calculates the mean force in that same interval.
- Displays the extracted curve and highlighted integration region.

## How it works

1. You provide a screenshot by either:
    - Paste (`Ctrl+V` / `Cmd+V`) directly into the app window, or
    - Upload an image file.
2. The app auto-detects chart axes and plot boundaries.
3. It finds the blue curve using color-dominance logic and connected-component tracking.
4. It calibrates pixel distances to mm and N.
5. It computes:

- Integrated area over the selected interval (displayed as mN*m)
- Mean force over the selected interval (N)
- It also writes extracted points to `xy-data.csv`.

## Example Screenshots

<img src="Screenshot%201.png" alt="Example screenshot of testXpert output" width="400">

<img src="Screenshot%20Output.png" alt="Example screenshot of the app with results" width="400">

## Run the app (Windows / PowerShell)

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install runtime dependencies:

```powershell
pip install dash plotly pandas numpy scipy pillow
```

Start the app:

```powershell
python app.py
```

Open in your browser:

```text
http://127.0.0.1:8050
```

## Development (optional)

Install development dependencies:

```powershell
pip install -r requirements-dev.txt
```

Install pre-commit hooks:

```powershell
pre-commit install
```

Run hooks manually on all files:

```powershell
pre-commit run --all-files
```

## Notes

- `APP_DEBUG` controls Dash debug mode (`true/false`, default: `true`).
- The current implementation is optimized for screenshots where the measured curve is blue.
