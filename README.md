# 🎯 SportsPredikt — MVP de Pronostics Sportifs en Temps Réel

Un tableau de bord Streamlit qui affiche les matchs du jour (Football & Basketball)
avec des probabilités de victoire calculées par des modèles ML, mises à jour en direct.

---

## 📋 Fonctionnalités

| Feature | Description |
|---|---|
| ⚽ **Football** | Premier League, La Liga, Ligue 1, Bundesliga, Serie A, Champions League, Coupe du Monde, CAN |
| 🏀 **Basketball** | Matchs NBA du jour |
| 📊 **Avant-match** | Elo + Poisson (foot) / Elo + Loi Normale (basket) |
| 🔴 **En direct** | Ajustement par score, minute, cartons rouges |
| 📈 **Graphique** | Évolution des probabilités pendant le match |
| 🔄 **Auto-refresh** | Mise à jour automatique toutes les 30 secondes |
| 💾 **Cache SQLite** | Limite les appels API (TTL configurable) |
| 🧪 **Backtest** | Walk-forward sur les 200 derniers matchs |

---

## 🚀 Installation

### 1. Cloner / télécharger le projet

```bash
cd bet/
```

### 2. Créer un environnement virtuel Python 3.11

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Configurer les clés API

```bash
cp .env.example .env
```

Éditez `.env` et renseignez :

```env
# Inscrivez-vous sur https://www.football-data.org/client/register
FOOTBALL_DATA_API_KEY=votre_clé_ici

# Balldontlie n'a pas besoin de clé API pour le plan gratuit
BALLDONTLIE_API_KEY=
```

### 5. Lancer l'application

```bash
streamlit run app.py
```

Ouvrez votre navigateur sur **http://localhost:8501**

---

## 🧪 Backtest

### Amorcer l'historique football

Les pronostics peuvent s'afficher sans base préremplie, mais leur entraînement et leur
calibration sont limités tant que l'application n'a pas accumulé de résultats.
Pour importer gratuitement des résultats historiques, téléchargez les CSV des ligues
souhaitées depuis [Football-Data.co.uk](https://www.football-data.co.uk/data.php)
(par exemple Premier League `E0`, La Liga `SP1`, Bundesliga `D1`, Serie A `I1`,
Ligue 1 `F1`), puis lancez depuis la racine du projet :

```bash
# Contrôler les fichiers sans écrire dans la base
python import_history.py --dry-run "C:\data\E0.csv" "C:\data\SP1.csv"

# Importer plusieurs saisons ou ligues (réimportable sans créer de doublons)
python import_history.py "C:\data\E0_2425.csv" "C:\data\E0_2526.csv" "C:\data\SP1_2526.csv"
```

Le format attendu est celui de Football-Data.co.uk (`Date`, `HomeTeam`, `AwayTeam`,
`FTHG`, `FTAG`). Les lignes sans score final sont ignorées ; les fichiers sont
importés dans la même base `cache.db` que celle utilisée par l'application.
Les cotes moyennes d'ouverture sont utilisées par le ROI simulé ; les cotes de
clôture présentes dans les CSV sont également conservées avec le match. Les
compétitions et saisons disponibles varient selon le site et le fichier.

Après l'import, relancez Streamlit : le modèle se réentraîne lorsque l'historique
change. Le modèle utilise au maximum ses 5 000 matchs les plus récents par sport.
Lancez ensuite le backtest pour vérifier la qualité prédictive avant de vous fier
aux probabilités.

Le backtest football affiche aussi un ROI simulé sur mises unitaires, à partir des
cotes moyennes d'ouverture importées et du seuil EV par défaut de 3 %. C'est une
simulation historique, pas une promesse de rendement ; les résultats dépendent
des fichiers et des cotes effectivement disponibles.

### Backtest

Après avoir importé ou accumulé des données :

```bash
# Football (200 matchs par défaut)
python backtest.py

# Basketball
python backtest.py --sport basket

# Les deux
python backtest.py --sport all

# Limiter à 50 matchs
python backtest.py --n 50
```

### Métriques retournées

| Métrique | Description |
|---|---|
| **Accuracy** | % de prédictions correctes (résultat le plus probable) |
| **Brier Score** | Erreur des probabilités (plus bas = mieux ; comparer à une baseline) |
| **ROI simulé** | Rendement historique à mise fixe sur les matchs avec cotes utilisables |

---

## 🏗️ Architecture

```
bet/
├── app.py              # Interface Streamlit principale
├── data_client.py      # Clients API (FootballClient, BasketballClient)
├── db.py               # Cache SQLite + persistance
├── import_history.py   # Import de résultats historiques Football-Data.co.uk
├── backtest.py         # Évaluation walk-forward des modèles
├── requirements.txt
├── .env.example        # Template des clés API
├── cache.db            # Base SQLite (auto-créée)
└── models/
    ├── __init__.py
    ├── elo.py          # Système de rating Elo
    ├── football.py     # Modèle Foot (Elo + Poisson)
    └── basketball.py   # Modèle Basket (Elo + Loi Normale)
```

---

## 🔬 Modèles Statistiques

### Football

**Avant-match** :
1. Le rating **Elo** de chaque équipe est calculé sur les matchs historiques
2. La différence de rating est convertie en **buts attendus** (λ home, λ away)
3. La distribution de **Poisson** itère sur tous les scores 0-0 à 10-10 pour calculer P(V), P(N), P(D)

**En direct** :
1. Les buts restants attendus sont proratisés selon le temps restant
2. L'ajustement des **cartons rouges** réduit le λ de l'équipe pénalisée (-15% par CR)
3. La distribution de Poisson est recalculée depuis le score actuel

### Basketball

**Avant-match** :
- Le différentiel Elo est converti en **spread de points** attendu
- P(victoire) = Φ(spread / σ) via la loi normale

**En direct** :
- σ restant = σ_base × √(minutes_restantes) — incertitude qui décroît avec le temps
- z = (spread_Elo + écart_score_actuel) / σ_restant

---

## ⚙️ Configuration avancée

### Limites API (plan gratuit)

| API | Limite |
|---|---|
| football-data.org | 10 req/min |
| balldontlie | 30 req/min (sans clé) |

Le cache SQLite est configuré pour limiter automatiquement les appels :
- Matchs du jour : 5 minutes
- Données en direct : 30 secondes
- Historique : 1 heure

### Ajuster le TTL du cache

Dans `db.py`, modifiez `CACHE_TTL` :

```python
CACHE_TTL = {
    "today_matches": 300,   # 5 minutes
    "live_match": 30,       # 30 secondes
    "history": 3600,        # 1 heure
}
```

---

## ☁️ Déploiement Cloud Gratuit (Streamlit Community Cloud)

Vous pouvez déployer cette application gratuitement pour y accéder 24h/24 depuis votre smartphone :

1. Rendez-vous sur [share.streamlit.io](https://share.streamlit.io/) et connectez-vous avec votre compte GitHub (**Callmexau**).
2. Cliquez sur **"New app"** et sélectionnez le dépôt **`Callmexau/bet`**, branche `main`, fichier `app.py`.
3. Cliquez sur **"Advanced settings"** > **Secrets** et collez vos clés d'environnement :
   ```toml
   FOOTBALL_DATA_API_KEY = "votre_cle"
   ODDS_API_KEY = "votre_cle"
   BALLDONTLIE_API_KEY = "votre_cle"
   TELEGRAM_BOT_TOKEN = "votre_token_optionnel"
   TELEGRAM_CHAT_ID = "votre_chat_id_optionnel"
   ```
4. Cliquez sur **Deploy** ! Votre application aura une URL publique HTTPS accessible partout.

---

## ⚠️ Avertissement légal

> Les probabilités affichées sont **indicatives** et basées sur des modèles statistiques.
> **Aucun résultat n'est garanti.** Cette application ne constitue pas un conseil en paris sportifs.

---

## 📚 Références

- [Dixon & Coles (1997)](https://www.jstor.org/stable/2988743) — Modélisation Poisson des matchs de football
- [Elo Rating System](https://en.wikipedia.org/wiki/Elo_rating_system)
- [Brier Score](https://en.wikipedia.org/wiki/Brier_score)
- [football-data.org API](https://www.football-data.org/documentation/quickstart)
- [balldontlie API](https://www.balldontlie.io/)
