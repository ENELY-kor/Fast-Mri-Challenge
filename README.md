# Fast-Mri-Challenge
The goal was to leverage CNNs to solve the problem of high cost and long scan times associated with MRI, specifically by accurately reconstructing images from subsampled data.
# Overview
Development of a CNN-based image reconstruction model to achieve the goal of the FastMRI Challenge (reducing MRI scan time). Focused on enhancing model robustness and generalization by incorporating the CutMix data augmentation technique.

**Key Tech Stack:** Python, CNN, CutMix, FastMRI
# Methodology & Implementation
* Initial Setup Studied relevant CNN coursework, performed Hyperparameter Tuning, and analyzed the base model architecture.

* Core Strategy - CutMix Applied CutMix (from '10-10 Project') to blend image regions and adjust labels. This was done to weaken the model's reliance on local feature dependency, forcing it to learn the overall context of visual information and improve generalization.
# Result

# Key Insights & Reflection
* Growth Identified a lack of deep knowledge as the cause of the performance ceiling, which motivated me to proactively pursue advanced studies (e.g., Graduate-level Computer System Programming, Information Theory), building strong Learning Agility.

* Technical Insight Realized that for simple, monochrome datasets, maximizing data augmentation (Flip, Rotate) and preventing overfitting is more critical and efficient than unnecessarily increasing model complexity.
