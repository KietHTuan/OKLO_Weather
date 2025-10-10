# 🌦️ NASA Space Apps 2025 – Weather Pattern Prediction

## 🎯 Goal
The goal of this project is to **predict weather patterns** using satellite-based atmospheric data provided by **NASA**.  
We aim to analyze multiple environmental parameters to classify different types of weather conditions based on historical patterns.

---

## 🛰️ Data Source
- **Source:** NASA GES DISC (Goddard Earth Sciences Data and Information Services Center)
- **Dataset:** AIRS/AMSU/HSB atmospheric data (Level 3)
- **Format:** `.nc4` (NetCDF) and `.csv`- after data extraction  
- **Variables include:**
  - Temperature
  - Humidity
  - CO₂ concentration
  - Longwave radiation
  - Latitude / Longitude / Time
  - Cloud and surface properties

---

## 🧠 Machine Learning Objective
This is a **multi-categorical classification problem** where the model predicts weather pattern categories Sunny, Rainy, Snowy


## Limitations 
The model precision only 82% there are more room for imporvement in future feature engineering process
