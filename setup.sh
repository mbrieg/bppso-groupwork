#!/usr/bin/env bash
set -e

# Erstellt eine Python-virtuelle Umgebung und installiert Abhängigkeiten
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "Fertig. Aktivieren Sie die Umgebung mit: source .venv/bin/activate"
echo "Starten Sie Jupyter Lab mit: jupyter lab"
