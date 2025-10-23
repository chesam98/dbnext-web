# dBNext — Analyse acoustique (v7.2) — Web (Render)

Portage **fidèle** de votre script bureau (Tkinter/Matplotlib) en **application web Dash** déployable sur **Render**.
- Import CSV dBNext/CESVA (colonnes: temps, LAeq)
- Paramétrage **Jour/Nuit** par défaut (et base pour extensions)
- **Nettoyage interactif**: gomme par sélection rect., seuil depuis clic, **Undo/Redo/Reset**
- **Grille X adaptative** + **traits de minuit** + lecture précise via *spikelines*
- **Export Excel** (openpyxl) avec **graphiques PNG** (tick minuit garanti + règle horaire en haut)
- Feuille par **point** et par **type** (LP / ZER / HYBRIDE) + **30 min caractéristiques**

## Lancer en local
```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python app.py  # http://localhost:8050
```

## Déploiement sur Render
1. Créez un dépôt Git avec ces fichiers (ou importez directement le ZIP).
2. Sur Render: **New → Web Service → Connect repository**.
3. Runtime: *Python*. Build: `pip install -r requirements.txt`. Start: `gunicorn app:server`.
4. Plan: *Free* (ou supérieur).
5. Déployez ▶️

## Parité fonctionnelle
- La logique **métier** (calculs LAeq/LN, rolling, intervalles, Excel/PNG) est **repris à l'identique** dans `dbnext_core.py`.
- L’édition web reproduit les outils:
  - **Gomme**: utilisez l’outil *Select* du graphique puis **🧹 Gomme**.
  - **Seuil**: cliquez sur le graphique après avoir pressé **🔺 Placer seuil** (optionnellement entrer un dB).
  - **Undo/Redo/Reset**: identique au bureau.
- Le **curseur croisé** est assuré par les **spikelines** Plotly (hover en `dd/mm HH:MM:SS` + dB).

> Remarque: les gestes clavier (Alt+drag / clic droit) ne sont pas fiables via navigateur ; ils sont remplacés par des commandes explicites équivalentes. Le **fonctionnement** et les **résultats** restent inchangés.

## Structure
```
.
├── app.py                 # Interface Dash + callbacks
├── dbnext_core.py         # Coeur métier + export Excel/PNG
├── requirements.txt
├── render.yaml            # Déploiement Render
├── Procfile               # (optionnel) gunicorn
└── assets/
    └── style.css
```
