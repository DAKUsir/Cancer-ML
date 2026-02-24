---
title: Cancer Risk Prediction
emoji: 🔬
colorFrom: red
colorTo: gray
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: mit
short_description: AI-powered cancer risk screening with 9 ML models, SHAP & LIME explanations
---

# Cancer Risk Prediction — AI Screening Tool

An AI-powered cancer risk screening tool using **9 machine learning models** with SHAP and LIME explanations.

## Features
- **9 Models**: Logistic L1/L2, Random Forest, Extra Trees, Gradient Boosting, SVM, KNN, and two ensemble models
- **SHAP** feature impact charts
- **LIME** local explanations
- **Single patient** and **batch CSV** prediction modes
- Tuned decision thresholds for high recall (catching cancer cases)

## Recommended Model
**HighRecall_Ensemble** — combines 4 models with boosted minority-class weights for the lowest false-negative rate.

## Disclaimer
This tool is for **screening purposes only** and does not constitute medical advice. Always consult a qualified healthcare professional.
