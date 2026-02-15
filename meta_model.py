# ==============================================================================
# meta_model.py
# ==============================================================================
# Machine Learning Meta-Model for Strategy Evaluation
#
# Trains classifiers/regressors on historical strategy performance to predict:
# 1. Survival probability (will strategy remain profitable?)
# 2. Overfitting risk score
# 3. FTMO pass probability
# 4. Expected out-of-sample Sharpe degradation
#
# Uses features from feature_engineering.py as input.
#
# Usage:
#     from meta_model import MetaModel
#     
#     model = MetaModel()
#     
#     # Train on historical data
#     model.train(feature_table, target='survived_6m')
#     
#     # Predict for new strategies
#     predictions = model.predict(new_features)
#     
#     # Get feature importance
#     importance = model.feature_importance()
#
# ==============================================================================

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from datetime import datetime
import warnings
import pickle
from pathlib import Path


@dataclass
class ModelMetrics:
    """Metrics from model training/evaluation"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: Optional[float]
    confusion_matrix: Optional[np.ndarray]
    cv_scores: List[float]
    cv_mean: float
    cv_std: float


@dataclass
class PredictionResult:
    """Result from model prediction"""
    strategy_name: str
    survival_probability: float
    overfitting_risk: float  # 0-1 scale
    confidence: float
    risk_factors: List[str]
    recommendation: str  # 'APPROVE', 'CAUTION', 'REJECT'


class MetaModel:
    """
    Machine Learning meta-model for strategy evaluation.
    
    Predicts strategy viability based on features extracted from
    backtesting, validation, and robustness testing.
    """
    
    def __init__(
        self,
        model_type: str = 'random_forest',
        random_state: int = 42
    ):
        """
        Args:
            model_type: 'random_forest', 'gradient_boosting', 'logistic', 'xgboost'
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.target_name = None
        self.is_fitted = False
        self.training_metrics = None
        
        # Default feature columns to use (subset of StrategyFeatures)
        self.default_features = [
            'total_return_pct',
            'sharpe_ratio',
            'max_drawdown_pct',
            'total_trades',
            'win_rate',
            'profit_factor',
            'trades_per_day',
            'avg_trade_duration_bars',
            'time_in_market_pct',
            'autocorr_lag1',
            'has_serial_dependence',
            'skewness',
            'kurtosis',
            'garch_persistence',
            'var_95_historical',
            'cvar_95',
            'latency_sensitivity',
            'combined_stress_survival',
            'net_return_pct',
            'cost_ratio'
        ]
    
    def _get_model(self):
        """Initialize the ML model"""
        
        if self.model_type == 'random_forest':
            try:
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_leaf=5,
                    random_state=self.random_state,
                    n_jobs=-1
                )
            except ImportError:
                raise ImportError("sklearn required. Install with: pip install scikit-learn")
        
        elif self.model_type == 'gradient_boosting':
            try:
                from sklearn.ensemble import GradientBoostingClassifier
                return GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=self.random_state
                )
            except ImportError:
                raise ImportError("sklearn required. Install with: pip install scikit-learn")
        
        elif self.model_type == 'logistic':
            try:
                from sklearn.linear_model import LogisticRegression
                return LogisticRegression(
                    max_iter=1000,
                    random_state=self.random_state
                )
            except ImportError:
                raise ImportError("sklearn required. Install with: pip install scikit-learn")
        
        elif self.model_type == 'xgboost':
            try:
                from xgboost import XGBClassifier
                return XGBClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=self.random_state,
                    use_label_encoder=False,
                    eval_metric='logloss'
                )
            except ImportError:
                raise ImportError("xgboost required. Install with: pip install xgboost")
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _prepare_features(
        self,
        df: pd.DataFrame,
        feature_cols: List[str] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """Prepare feature matrix from DataFrame"""
        
        if feature_cols is None:
            feature_cols = [c for c in self.default_features if c in df.columns]
        
        X = df[feature_cols].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Convert booleans to int
        for col in X.columns:
            if X[col].dtype == bool:
                X[col] = X[col].astype(int)
        
        return X.values, list(X.columns)
    
    def train(
        self,
        feature_df: pd.DataFrame,
        target: str,
        feature_cols: List[str] = None,
        test_size: float = 0.2,
        cv_folds: int = 5
    ) -> ModelMetrics:
        """
        Train the meta-model on historical strategy data.
        
        Args:
            feature_df: DataFrame with strategy features
            target: Column name for target variable (e.g., 'survived_6m', 'is_overfit')
            feature_cols: List of feature columns to use (default: all numeric)
            test_size: Fraction for test split
            cv_folds: Number of cross-validation folds
        
        Returns:
            ModelMetrics with training results
        """
        
        try:
            from sklearn.model_selection import train_test_split, cross_val_score
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score, 
                f1_score, roc_auc_score, confusion_matrix
            )
        except ImportError:
            raise ImportError("sklearn required. Install with: pip install scikit-learn")
        
        print(f"\n{'='*60}")
        print(f"TRAINING META-MODEL")
        print(f"{'='*60}")
        print(f"Model Type: {self.model_type}")
        print(f"Target: {target}")
        print(f"Samples: {len(feature_df)}")
        
        # Prepare features
        X, feature_names = self._prepare_features(feature_df, feature_cols)
        y = feature_df[target].values
        
        print(f"Features: {len(feature_names)}")
        print(f"Target distribution: {np.bincount(y.astype(int))}")
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Initialize and train model
        self.model = self._get_model()
        self.model.fit(X_train, y_train)
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1] if hasattr(self.model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=cv_folds)
        
        # Store state
        self.feature_names = feature_names
        self.target_name = target
        self.is_fitted = True
        
        metrics = ModelMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_roc=auc,
            confusion_matrix=conf_matrix,
            cv_scores=list(cv_scores),
            cv_mean=cv_scores.mean(),
            cv_std=cv_scores.std()
        )
        
        self.training_metrics = metrics
        
        # Print results
        print(f"\n{'─'*60}")
        print(f"TRAINING RESULTS:")
        print(f"{'─'*60}")
        print(f"  Accuracy:    {accuracy:.3f}")
        print(f"  Precision:   {precision:.3f}")
        print(f"  Recall:      {recall:.3f}")
        print(f"  F1 Score:    {f1:.3f}")
        if auc:
            print(f"  AUC-ROC:     {auc:.3f}")
        print(f"  CV Mean:     {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
        print(f"{'='*60}")
        
        return metrics
    
    def predict(
        self,
        features: Union[pd.DataFrame, Dict]
    ) -> List[PredictionResult]:
        """
        Predict survival/viability for strategies.
        
        Args:
            features: DataFrame or dict with strategy features
        
        Returns:
            List of PredictionResult
        """
        
        if not self.is_fitted:
            raise ValueError("Model not trained. Call train() first.")
        
        # Convert dict to DataFrame if needed
        if isinstance(features, dict):
            features = pd.DataFrame([features])
        
        results = []
        
        for idx, row in features.iterrows():
            # Prepare features
            X, _ = self._prepare_features(pd.DataFrame([row]), self.feature_names)
            X_scaled = self.scaler.transform(X)
            
            # Get prediction and probability
            pred = self.model.predict(X_scaled)[0]
            
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(X_scaled)[0, 1]
            else:
                proba = float(pred)
            
            # Identify risk factors
            risk_factors = self._identify_risk_factors(row)
            
            # Calculate overfitting risk
            overfit_risk = self._calculate_overfit_risk(row)
            
            # Determine recommendation
            if proba >= 0.7 and overfit_risk < 0.3:
                recommendation = 'APPROVE'
            elif proba >= 0.4 or overfit_risk < 0.5:
                recommendation = 'CAUTION'
            else:
                recommendation = 'REJECT'
            
            results.append(PredictionResult(
                strategy_name=row.get('strategy_name', f'Strategy_{idx}'),
                survival_probability=proba,
                overfitting_risk=overfit_risk,
                confidence=1 - abs(proba - 0.5) * 2,  # Higher near 0 or 1
                risk_factors=risk_factors,
                recommendation=recommendation
            ))
        
        return results
    
    def _identify_risk_factors(self, row: pd.Series) -> List[str]:
        """Identify risk factors for a strategy"""
        
        risk_factors = []
        
        # Check various thresholds
        if row.get('total_trades', 0) < 30:
            risk_factors.append("Low trade count (<30)")
        
        if row.get('has_serial_dependence', False):
            risk_factors.append("Serial dependence in returns")
        
        skew = row.get('skewness', 0)
        if skew < -1:
            risk_factors.append(f"Negative skew ({skew:.2f}) - crash risk")
        
        kurt = row.get('kurtosis', 0)
        if kurt > 3:
            risk_factors.append(f"High kurtosis ({kurt:.2f}) - fat tails")
        
        if row.get('max_drawdown_pct', 0) > 20:
            risk_factors.append("High max drawdown (>20%)")
        
        if row.get('cost_ratio', 0) > 50:
            risk_factors.append("High cost ratio (>50%)")
        
        garch = row.get('garch_persistence', 0)
        if garch > 0.95:
            risk_factors.append("High volatility persistence")
        
        if row.get('latency_sensitivity', 0) > 2:
            risk_factors.append("Latency sensitive")
        
        sharpe = row.get('sharpe_ratio', 0)
        if sharpe and sharpe > 3:
            risk_factors.append("Suspiciously high Sharpe (>3)")
        
        return risk_factors
    
    def _calculate_overfit_risk(self, row: pd.Series) -> float:
        """Calculate overfitting risk score (0-1)"""
        
        risk_score = 0.0
        factors = 0
        
        # Low trade count
        trades = row.get('total_trades', 0)
        if trades < 100:
            risk_score += 0.3 * (1 - trades / 100)
            factors += 1
        
        # Serial correlation (inflates significance)
        if row.get('has_serial_dependence', False):
            risk_score += 0.2
            factors += 1
        
        # Extreme Sharpe
        sharpe = row.get('sharpe_ratio', 0)
        if sharpe and sharpe > 2.5:
            risk_score += 0.2 * min(1, (sharpe - 2.5) / 1.5)
            factors += 1
        
        # Non-normal distribution (parametric tests invalid)
        if not row.get('is_normal_distribution', True):
            risk_score += 0.1
            factors += 1
        
        # High cost ratio (edge may be illusory)
        if row.get('cost_ratio', 0) > 30:
            risk_score += 0.1
            factors += 1
        
        # Low robustness
        if row.get('combined_stress_survival', 100) < 50:
            risk_score += 0.2
            factors += 1
        
        return min(1.0, risk_score)
    
    def feature_importance(self) -> pd.DataFrame:
        """Get feature importance from trained model"""
        
        if not self.is_fitted:
            raise ValueError("Model not trained. Call train() first.")
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_[0])
        else:
            return pd.DataFrame()
        
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df
    
    def save_model(self, path: str):
        """Save trained model to file"""
        
        if not self.is_fitted:
            raise ValueError("Model not trained. Call train() first.")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'model_type': self.model_type,
            'training_metrics': self.training_metrics
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model from file"""
        
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.target_name = model_data['target_name']
        self.model_type = model_data['model_type']
        self.training_metrics = model_data['training_metrics']
        self.is_fitted = True
        
        print(f"Model loaded from {path}")
    
    def print_prediction_report(self, predictions: List[PredictionResult]):
        """Print formatted prediction report"""
        
        print("\n" + "="*70)
        print("META-MODEL PREDICTIONS")
        print("="*70)
        
        for pred in predictions:
            print(f"\n📊 {pred.strategy_name}")
            print("-"*70)
            print(f"  Survival Probability:  {pred.survival_probability*100:>6.1f}%")
            print(f"  Overfitting Risk:      {pred.overfitting_risk*100:>6.1f}%")
            print(f"  Confidence:            {pred.confidence*100:>6.1f}%")
            
            rec_icon = "✅" if pred.recommendation == 'APPROVE' else "⚠️" if pred.recommendation == 'CAUTION' else "❌"
            print(f"  Recommendation:        {rec_icon} {pred.recommendation}")
            
            if pred.risk_factors:
                print(f"  Risk Factors:")
                for factor in pred.risk_factors:
                    print(f"    • {factor}")
        
        print("\n" + "="*70)


# ==============================================================================
# EARLY KILL FILTER
# ==============================================================================

class EarlyKillFilter:
    """
    Quick filter to reject obviously bad strategies before expensive validation.
    
    Uses simple heuristics, not ML.
    """
    
    def __init__(
        self,
        min_trades: int = 20,
        min_sharpe: float = 0.0,
        max_drawdown: float = 30.0,
        min_win_rate: float = 30.0,
        max_cost_ratio: float = 80.0
    ):
        self.min_trades = min_trades
        self.min_sharpe = min_sharpe
        self.max_drawdown = max_drawdown
        self.min_win_rate = min_win_rate
        self.max_cost_ratio = max_cost_ratio
    
    def should_kill(self, backtest_result: Dict) -> Tuple[bool, List[str]]:
        """
        Check if strategy should be killed early.
        
        Returns:
            (should_kill, reasons)
        """
        reasons = []
        
        # Check trade count
        trades = backtest_result.get('total_trades', 0)
        if trades < self.min_trades:
            reasons.append(f"Too few trades ({trades} < {self.min_trades})")
        
        # Check Sharpe
        sharpe = backtest_result.get('sharpe_ratio')
        if sharpe is not None and sharpe < self.min_sharpe:
            reasons.append(f"Low Sharpe ({sharpe:.2f} < {self.min_sharpe})")
        
        # Check drawdown
        dd = backtest_result.get('max_drawdown_pct', 0)
        if dd > self.max_drawdown:
            reasons.append(f"High drawdown ({dd:.1f}% > {self.max_drawdown}%)")
        
        # Check win rate
        wr = backtest_result.get('win_rate')
        if wr is not None and wr < self.min_win_rate:
            reasons.append(f"Low win rate ({wr:.1f}% < {self.min_win_rate}%)")
        
        # Check if negative return
        ret = backtest_result.get('total_return_pct', 0)
        if ret < 0:
            reasons.append(f"Negative return ({ret:.2f}%)")
        
        return len(reasons) > 0, reasons


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def quick_predict(features: pd.DataFrame, model_path: str = None) -> List[PredictionResult]:
    """Quick prediction using saved or new model"""
    model = MetaModel()
    if model_path and Path(model_path).exists():
        model.load_model(model_path)
    else:
        warnings.warn("No trained model available. Using heuristics only.")
        # Return heuristic-based predictions
        results = []
        for idx, row in features.iterrows():
            risk = model._calculate_overfit_risk(row)
            results.append(PredictionResult(
                strategy_name=row.get('strategy_name', f'Strategy_{idx}'),
                survival_probability=1 - risk,
                overfitting_risk=risk,
                confidence=0.5,
                risk_factors=model._identify_risk_factors(row),
                recommendation='CAUTION'
            ))
        return results
    
    return model.predict(features)


# ==============================================================================
# MAIN (Testing)
# ==============================================================================

if __name__ == "__main__":
    print("="*70)
    print("META-MODEL TEST")
    print("="*70)
    
    # Create sample training data
    np.random.seed(42)
    n_samples = 200
    
    # Generate synthetic features
    data = {
        'strategy_name': [f'Strategy_{i}' for i in range(n_samples)],
        'total_return_pct': np.random.normal(10, 15, n_samples),
        'sharpe_ratio': np.random.normal(1.0, 0.8, n_samples),
        'max_drawdown_pct': np.abs(np.random.normal(10, 8, n_samples)),
        'total_trades': np.random.randint(20, 200, n_samples),
        'win_rate': np.random.normal(50, 10, n_samples),
        'profit_factor': np.random.normal(1.5, 0.5, n_samples),
        'trades_per_day': np.random.uniform(0.1, 2, n_samples),
        'avg_trade_duration_bars': np.random.randint(5, 50, n_samples),
        'time_in_market_pct': np.random.uniform(10, 80, n_samples),
        'autocorr_lag1': np.random.normal(0, 0.1, n_samples),
        'has_serial_dependence': np.random.choice([True, False], n_samples, p=[0.2, 0.8]),
        'skewness': np.random.normal(0, 0.5, n_samples),
        'kurtosis': np.random.normal(0, 1, n_samples),
        'garch_persistence': np.random.uniform(0.5, 0.99, n_samples),
        'var_95_historical': np.random.normal(-2, 1, n_samples),
        'cvar_95': np.random.normal(-3, 1.5, n_samples),
        'latency_sensitivity': np.random.uniform(0, 3, n_samples),
        'combined_stress_survival': np.random.uniform(20, 100, n_samples),
        'net_return_pct': np.random.normal(8, 15, n_samples),
        'cost_ratio': np.random.uniform(5, 60, n_samples),
    }
    
    # Create synthetic target (survived 6 months)
    # Higher Sharpe, lower drawdown, more trades = more likely to survive
    survival_score = (
        data['sharpe_ratio'] * 0.3 +
        (100 - data['max_drawdown_pct']) * 0.02 +
        np.log(data['total_trades']) * 0.1 -
        data['cost_ratio'] * 0.01
    )
    data['survived_6m'] = (survival_score + np.random.normal(0, 0.5, n_samples) > 1.5).astype(int)
    
    df = pd.DataFrame(data)
    
    print(f"\nSample data shape: {df.shape}")
    print(f"Target distribution: {df['survived_6m'].value_counts().to_dict()}")
    
    # Train model
    model = MetaModel(model_type='random_forest')
    metrics = model.train(df, target='survived_6m')
    
    # Feature importance
    print("\nFeature Importance:")
    importance = model.feature_importance()
    print(importance.head(10).to_string(index=False))
    
    # Test prediction
    print("\nTesting predictions...")
    test_strategies = df.head(3)
    predictions = model.predict(test_strategies)
    model.print_prediction_report(predictions)
    
    # Test early kill filter
    print("\nTesting Early Kill Filter...")
    killer = EarlyKillFilter()
    
    bad_strategy = {'total_trades': 10, 'sharpe_ratio': -0.5, 'max_drawdown_pct': 35}
    should_kill, reasons = killer.should_kill(bad_strategy)
    print(f"Bad strategy - Kill: {should_kill}, Reasons: {reasons}")
    
    good_strategy = {'total_trades': 100, 'sharpe_ratio': 1.5, 'max_drawdown_pct': 10}
    should_kill, reasons = killer.should_kill(good_strategy)
    print(f"Good strategy - Kill: {should_kill}, Reasons: {reasons}")
    
    print("\n" + "="*70)
    print("Meta-model working!")
    print("="*70)