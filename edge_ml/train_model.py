#!/usr/bin/env python3
"""
Train a DecisionTreeClassifier on synthetic soil sensor data
and export the model as a raw C header file using micromlgen.

This script:
1. Generates synthetic soil readings (moisture, nitrogen, phosphorus, potassium, temperature)
2. Creates labels for irrigation_needed based on soil conditions
3. Trains a scikit-learn DecisionTreeClassifier
4. Exports the model as a microml-compliant C header file (model.h)
"""

from xml.parsers.expat import model

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from micromlgen import port


def generate_synthetic_soil_data(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic soil sensor readings.

    Features:
        - moisture: Soil moisture content (0-100%)
        - nitrogen: Nitrogen level (0-200 ppm)
        - phosphorus: Phosphorus level (0-100 ppm)
        - potassium: Potassium level (0-200 ppm)
        - temperature: Soil temperature (0-45°C)

    Returns:
        DataFrame with features and irrigation_needed label
    """
    np.random.seed(random_state)

    # Generate feature distributions based on realistic soil sensor ranges
    moisture = np.clip(
        np.random.normal(loc=45, scale=12, size=n_samples),
        0, 100
    )

    nitrogen = np.clip(
        np.random.normal(loc=80, scale=30, size=n_samples),
        0, 200
    )

    phosphorus = np.clip(
        np.random.normal(loc=40, scale=15, size=n_samples),
        0, 100
    )

    potassium = np.clip(
        np.random.normal(loc=120, scale=40, size=n_samples),
        0, 200
    )

    temperature = np.clip(
        np.random.normal(loc=22, scale=8, size=n_samples),
        0, 45
    )

    # Create irrigation_needed label based on soil conditions
    # Irrigation is needed when moisture is low OR multiple nutrients are depleted
    low_moisture = moisture < 35
    low_nitrogen = nitrogen < 60
    low_phosphorus = phosphorus < 30
    low_potassium = potassium < 90

    # Irrigation needed if:
    # - Moisture is critically low, OR
    # - Multiple nutrients are depleted (stress condition)
    irrigation_needed = (
        low_moisture |
        ((low_nitrogen & low_phosphorus) | (low_nitrogen & low_potassium))
    )

    df = pd.DataFrame({
        'moisture': moisture,
        'nitrogen': nitrogen,
        'phosphorus': phosphorus,
        'potassium': potassium,
        'temperature': temperature,
        'irrigation_needed': irrigation_needed.astype(int)
    })

    return df


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> DecisionTreeClassifier:
    """
    Train a DecisionTreeClassifier on the provided data.

    Args:
        X_train: Training features (n_samples, n_features)
        y_train: Training labels (n_samples,)

    Returns:
        Trained DecisionTreeClassifier model
    """
    # Use max_depth=4 to keep the tree simple and interpretable for edge deployment
    # This also helps with model size for C export
    model = DecisionTreeClassifier(
        max_depth=4,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight='balanced'  # Handle potential class imbalance
    )

    model.fit(X_train, y_train)
    return model


def evaluate_model(model: DecisionTreeClassifier, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate the trained model on test data.

    Args:
        model: Trained classifier
        X_test: Test features
        y_test: Test labels

    Returns:
        Dictionary with evaluation metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    y_pred = model.predict(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, pos_label=1),
        'recall': recall_score(y_test, y_pred, pos_label=1),
        'f1_score': f1_score(y_test, y_pred, pos_label=1)
    }

    return metrics


def main():
    """Main training and export pipeline."""
    print("=" * 60)
    print("Soil Irrigation Prediction Model Training")
    print("=" * 60)

    # Step 1: Generate synthetic dataset
    print("\n[1/5] Generating synthetic soil sensor data...")
    df = generate_synthetic_soil_data(n_samples=1000, random_state=42)
    print(f"Generated {len(df)} samples")
    print(f"Features: moisture, nitrogen, phosphorus, potassium, temperature")
    print(f"Class distribution:")
    print(f"- irrigation_needed=False: {(df['irrigation_needed'] == 0).sum()} ({100*df['irrigation_needed'].mean():.1f}%)")
    print(f"- irrigation_needed=True: {(df['irrigation_needed'] == 1).sum()} ({100*(1-df['irrigation_needed']).mean():.1f}%)")

    # Step 2: Prepare features and labels
    feature_columns = ['moisture', 'nitrogen', 'phosphorus', 'potassium', 'temperature']
    X = df[feature_columns].values
    y = df['irrigation_needed'].values

    # Split into train/test sets (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n[2/5] Data split: {len(X_train)} training samples, {len(X_test)} test samples")

    # Step 3: Train the model
    print("\n[3/5] Training DecisionTreeClassifier...")
    model = train_model(X_train, y_train)
    print(f"Tree depth: {model.get_depth()}")
    print(f"Number of leaves: {model.get_n_leaves()}")
    print(f"Number of features: {model.n_features_in_}")
    # Step 4: Evaluate on test set
    print("\n[4/5] Evaluating model on test set...")
    metrics = evaluate_model(model, X_test, y_test)
    print(f"Accuracy:   {metrics['accuracy']:.3f}")
    print(f"Precision:  {metrics['precision']:.3f}")
    print(f"Recall:     {metrics['recall']:.3f}")
    print(f"F1 Score:   {metrics['f1_score']:.3f}")

    # Step 5: Export model as C header file
    print("\n[5/5] Exporting model to C header file...")
    header_path = 'model.h'
    c_code = port(model)
    with open(header_path, 'w') as file:
        file.write(c_code)
    print(f"Model exported to: {header_path}")

    # Display the generated header file (first 50 lines)
    print("\n" + "=" * 60)
    print("Generated C Header File Preview:")
    print("=" * 60)
    with open(header_path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines[:50]):
            print(f"{i+1:3d}: {line}", end='')

    print("\n" + "=" * 60)
    print("Training and export completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
