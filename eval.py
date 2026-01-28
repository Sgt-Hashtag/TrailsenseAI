import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

def run_ml_evaluation(model_path="trailsense_model.pkl"):
    """
    Comprehensive evaluation of the TrailSense suitability model.
    Tests the model's ability to predict athlete route preferences.
    """
    
    #load the trained model
    print("Loading model from:", model_path)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    #validation set based on Berlin
    np.random.seed(42)
    n_samples = 100
    
    test_scenarios = []
    
    # Urban roads (30%)
    urban = pd.DataFrame({
        'incline': np.random.uniform(0, 0.05, 30),
        'is_trail': [0] * 30,
        'greenery': np.random.uniform(0, 0.3, 30),
        'type': ['urban'] * 30
    })
    
    # Parks & trails (40%)
    parks = pd.DataFrame({
        'incline': np.random.uniform(0, 0.08, 40),
        'is_trail': [1] * 40,
        'greenery': np.random.uniform(0.6, 1.0, 40),
        'type': ['park'] * 40
    })
    
    # Hilly areas (20%)
    hilly = pd.DataFrame({
        'incline': np.random.uniform(0.08, 0.25, 20),
        'is_trail': np.random.randint(0, 2, 20),
        'greenery': np.random.uniform(0.3, 0.7, 20),
        'type': ['hilly'] * 20
    })

    # Waterfront paths (10%)
    waterfront = pd.DataFrame({
        'incline': np.random.uniform(0, 0.02, 10),
        'is_trail': [1] * 10,
        'greenery': np.random.uniform(0.5, 0.9, 10),
        'type': ['waterfront'] * 10
    })
    
    df_test = pd.concat([urban, parks, hilly, waterfront], ignore_index=True)
    
    # Ground Truth based on athlete preferences: trails > roads, greenery > concrete, moderate incline > flat or steep
    def calculate_true_popularity(row):
        score = 0.5  # baseline
        
        # Trail bonus
        score += row['is_trail'] * 0.3
        
        # Greenery bonus
        score += row['greenery'] * 0.25
        
        # Incline preference
        if 0.05 <= row['incline'] <= 0.10:
            score += 0.15  # optimal incline
        elif row['incline'] > 0.15:
            score -= 0.20  # too steep
        elif row['incline'] < 0.02:
            score -= 0.05  # too flat
        
        return np.clip(score, 0, 1)
    
    y_true = df_test.apply(calculate_true_popularity, axis=1)

    # Perform Inference
    X_test = df_test[['incline', 'is_trail', 'greenery']]
    y_pred = model.predict(X_test)

    # Quantitative Metrics
    metrics = {
        "MAE (Mean Absolute Error)": mean_absolute_error(y_true, y_pred),
        "RMSE (Root Mean Squared Error)": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R² Score (Coefficient of Determination)": r2_score(y_true, y_pred),
        "Mean Prediction": y_pred.mean(),
        "Std Prediction": y_pred.std()
    }

    print("\n" + "="*50)
    print("MODEL PERFORMANCE METRICS")
    print("="*50)
    for k, v in metrics.items():
        print(f"{k:40s}: {v:.4f}")
    
    # Interpretation guide
    print("\n Interpretation Guide:")
    print(f"  • MAE < 0.15: Excellent | 0.15-0.25: Good | >0.25: Needs improvement")
    print(f"  • R² > 0.70: Strong fit | 0.50-0.70: Moderate | <0.50: Weak")
    
    #Scenario-based Analysis
    print("\n" + "="*50)
    print("SCENARIO BREAKDOWN")
    print("="*50)
    
    for scenario in ['urban', 'park', 'hilly', 'waterfront']:
        mask = df_test['type'] == scenario
        if mask.sum() > 0:
            scenario_mae = mean_absolute_error(y_true[mask], y_pred[mask])
            scenario_r2 = r2_score(y_true[mask], y_pred[mask])
            print(f"{scenario.capitalize():15s}: MAE={scenario_mae:.3f}, R²={scenario_r2:.3f}")

    #Visualization: Predicted vs Actual
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('TrailSense AI: Model Evaluation Dashboard', fontsize=16, fontweight='bold')
    
    # Plot 1: Regression plot
    ax1 = axes[0, 0]
    sns.regplot(x=y_true, y=y_pred, ax=ax1, scatter_kws={'alpha':0.6, 's': 50}, 
                line_kws={'color':'#FC4C02', 'linewidth': 2})
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect Prediction')
    ax1.set_title("Predicted vs. Ground Truth Popularity")
    ax1.set_xlabel("Ground Truth Popularity")
    ax1.set_ylabel("ML Predicted Popularity")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # Plot 2: Residuals
    ax2 = axes[0, 1]
    residuals = y_pred - y_true
    ax2.scatter(y_pred, residuals, alpha=0.6, s=50)
    ax2.axhline(y=0, color='#FC4C02', linestyle='--', linewidth=2)
    ax2.set_title("Residual Plot")
    ax2.set_xlabel("Predicted Popularity")
    ax2.set_ylabel("Residuals (Predicted - Actual)")
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    # Plot 3: Distribution by scenario
    ax3 = axes[1, 0]
    scenario_data = []
    for scenario in ['urban', 'park', 'hilly', 'waterfront']:
        mask = df_test['type'] == scenario
        scenario_data.append(y_pred[mask])
    
    ax3.boxplot(scenario_data, tick_labels=['Urban', 'Park', 'Hilly', 'Waterfront'])
    ax3.set_title("Predicted Popularity by Scenario")
    ax3.set_ylabel("Predicted Popularity Score")
    ax3.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    # Plot 4: Feature Importance
    ax4 = axes[1, 1]
    importances = pd.Series(model.feature_importances_,index=['Incline', 'Is Trail', 'Greenery'])
    importances.sort_values().plot(kind='barh', ax=ax4, color='#FC4C02')
    ax4.set_title("Feature Importance")
    ax4.set_xlabel("Importance Score")
    ax4.grid(True, linestyle='--', alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig("trailsense_model_evaluation.png", dpi=300, bbox_inches='tight')
    print("\n saved to: trailsense_model_evaluation.png")
    
    # Feature Importance Details
    print("\n" + "="*50)
    print("FEATURE IMPORTANCE RANKING")
    print("="*50)
    for feature, importance in importances.sort_values(ascending=False).items():
        print(f"{feature:15s}: {importance:.4f} ({importance*100:.1f}%)")
    
    # Edge Case Testing
    print("\n" + "="*50)
    print("EDGE CASE TESTING")
    print("="*50)
    
    edge_cases = pd.DataFrame({
        'incline': [0.0, 0.25, 0.10, 0.0, 0.15],
        'is_trail': [0, 0, 1, 1, 1],
        'greenery': [0.0, 0.0, 1.0, 1.0, 0.5],
        'description': [
            'Flat urban road (least popular)',
            'Steep urban road (challenging)',
            'Perfect trail (most popular)',
            'Flat trail with greenery',
            'Moderate hilly trail'
        ]
    })
    
    edge_predictions = model.predict(edge_cases[['incline', 'is_trail', 'greenery']])
    
    for i, row in edge_cases.iterrows():
        print(f"{row['description']:35s}: {edge_predictions[i]:.3f}")
    
    print("\n" + "="*50)
    
    if metrics["R² Score (Coefficient of Determination)"] > 0.7:
        print("Model good")
    elif metrics["R² Score (Coefficient of Determination)"] > 0.5:
        print("Model moderate more training data")
    else:
        print("Model needs improvement with diverse examples")
    
    if metrics["MAE (Mean Absolute Error)"] > 0.2:
        print(" High error rate - model may benefit from: More training samples, Additional features, Different algorithm viz boosting methods")
    
    return metrics, df_test, y_true, y_pred

if __name__ == "__main__":
    metrics, test_data, y_true, y_pred = run_ml_evaluation()
    
    print("\n" + "="*50)
    print("Eval Complete!")
    print("="*50)