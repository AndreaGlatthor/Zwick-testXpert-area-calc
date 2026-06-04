# Zwick-testXpert-area-calc

## Setup (Windows / PowerShell)

1. Virtuelle Umgebung erstellen:

```powershell
python -m venv .venv
```

2. Virtuelle Umgebung aktivieren:

```powershell
.\.venv\Scripts\Activate.ps1
```

3. Laufzeit-Abhaengigkeiten installieren:

```powershell
pip install dash plotly pandas numpy scipy pillow
```

4. Dev-Abhaengigkeiten installieren:

```powershell
pip install -r requirements-dev.txt
```

## Pre-commit einrichten

Einmalig im Repository ausfuehren:

```powershell
pre-commit install
```

Optional: alle Hooks manuell auf den gesamten Stand laufen lassen:

```powershell
pre-commit run --all-files
```

## App starten

Mit aktivierter venv:

```powershell
python app.py
```

Dann im Browser aufrufen: http://127.0.0.1:8050