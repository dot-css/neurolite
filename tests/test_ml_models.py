"""
Unit tests for traditional ML model implementations.

Tests scikit-learn, XGBoost, and ensemble model implementations.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from neurolite.models.base import BaseModel, TaskType, ModelCapabilities, PredictionResult
from neurolite.models.ml.sklearn_models import (
    create_random_forest_classifier,
    create_random_forest_regressor,
    create_logistic_regression,
    create_linear_regression,
    create_svm_classifier,
    create_gradient_boosting_classifier,
    create_gradient_boosting_regressor,
    register_sklearn_models
)
from neurolite.models.ml.xgboost_models import (
    create_xgboost_classifier,
    create_xgboost_regressor,
    register_xgboost_models
)
from neurolite.models.ml.ensemble_models import (
    VotingEnsemble,
    StackingEnsemble,
    create_voting_classifier,
    create_voting_regressor,
    create_stacking_classifier,
    create_stacking_regressor,
    register_ensemble_models
)
from neurolite.data.detector import DataType
from neurolite.core.exceptions import DependencyError, ModelError
from neurolite.models.registry import get_model_registry


class TestSklearnModels:
    """Test scikit-learn model implementations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create sample data
        np.random.seed(42)
        self.X_classification = np.random.rand(100, 4)
        self.y_classification = np.random.randint(0, 3, 100)
        
        self.X_regression = np.random.rand(100, 4)
        self.y_regression = np.random.rand(100) * 10
        
        # Small dataset for quick testing
        self.X_small = np.random.rand(20, 4)
        self.y_small_class = np.random.randint(0, 2, 20)
        self.y_small_reg = np.random.rand(20) * 10
    
    @pytest.mark.skipif(
        not pytest.importorskip("sklearn", reason="scikit-learn not available"),
        reason="scikit-learn not available"
    )
    def test_random_forest_classifier_creation(self):
        """Test Random Forest classifier creation."""
        model = create_random_forest_classifier(n_estimators=10)
        
        assert isinstance(model, BaseModel)
        assert TaskType.CLASSIFICATION in model.capabilities.supported_tasks
        assert DataType.TABULAR in model.capabilities.supported_data_types
        assert model.capabilities.framework == "sklearn"
        assert model.capabilities.supports_feature_importance
        assert model.capabilities.supports_probability_prediction
    
    @pytest.mark.skipif(
        not pytest.importorskip("sklearn", reason="scikit-learn not available"),
        reason="scikit-learn not available"
    )
    def test_random_forest_classifier_training(self):
        """Test Random Forest classifier training and prediction."""
        model = create_random_forest_classifier(n_estimators=10)
        
        # Train model
        model.fit(self.X_small, self.y_small_class)
        assert model.is_trained
        
        # Make predictions
        result = model.predict(self.X_small[:5])
        assert isinstance(result, PredictionResult)
        assert result.predictions.shape == (5,)
        assert result.probabilities is not None
        assert result.probabilities.shape == (5, 2)  # Binary classification
        assert result.feature_importance is not None
        assert len(result.feature_importance) == 4  # Number of features
    
    @pytest.mark.skipif(
        not pytest.importorskip("sklearn", reason="scikit-learn not available"),
        reason="scikit-learn not available"
    )
    def test_random_forest_regressor_creation(self):
        """Test Random Forest regressor creation."""
        model = create_random_forest_regressor(n_estimators=10)
        
        assert isinstance(model, BaseModel)
        assert TaskType.REGRESSION in model.capabilities.supported_tasks
        assert DataType.TABULAR in model.capabilities.supported_data_types
        assert model.capabilities.framework == "sklearn"
        assert model.capabilities.supports_feature_importance
    
    @pytest.mark.skipif(
        not pytest.importorskip("sklearn", reason="scikit-learn not available"),
        reason="scikit-learn not available"
    )
    def test_random_forest_regressor_training(self):
        """Test Random Forest regressor training and prediction."""
        model = create_random_forest_regressor(n_estimators=10)
        
        # Train model
        model.fit(self.X_small, self.y_small_reg)
        assert model.is_trained
        
        # Make predictions
        result = model.predict(self.X_small[:5])
        assert isinstance(result, PredictionResult)
        assert result.predictions.shape == (5,)
        assert result.probabilities is None  # Regression doesn't have probabilities
        assert result.feature_importance is not None
    
    @pytest.mark.skipif(
        not pytest.importorskip("sklearn", reason="scikit-learn not available"),
        reason="scikit-learn not available"
    )
    def test_logistic_regression_creation(self):
        """Test Logistic Regression creation."""
        model = create_logistic_regression()
        
        assert isinstance(model, BaseModel)
        assert TaskType.CLASSIFICATION in model.capabilities.supported_tasks
        assert model.capabilities.supports_probability_prediction
    
    @pytest.mark.skipif(
        not pytest.importorskip("sklearn", reason="scikit-learn not available"),
        reason="scikit-learn not available"
    )
    def test_logistic_regression_training(self):
        """Test Logistic Regression training and prediction."""
        model = create_logistic_regression()
        
        # Train model
        model.fit(self.X_small, self.y_small_class)
        assert model.is_trained
        
        # Make predictions
        result = model.predict(self.X_small[:5])
        assert isinstance(result, PredictionResult)
        assert result.predictions.shape == (5,)
        assert result.probabilities is not None
    
    @pytest.mark.skipif(
        not pytest.importorskip("sklearn", reason="scikit-learn not available"),
        reason="scikit-learn not available"
    )
    def test_linear_regression_creation(self):
        """Test Linear Regression creation."""
        model = create_linear_regression()
        
        assert isinstance(model, BaseModel)
        assert TaskType.REGRESSION in model.capabilities.supported_tasks
        assert not model.capabilities.supports_probability_prediction
    
    @pytest.mark.skipif(
        not pytest.importorskip("sklearn", reason="scikit-learn not available"),
        reason="scikit-learn not available"
    )
    def test_linear_regression_training(self):
        """Test Linear Regression training and prediction."""
        model = create_linear_regression()
        
        # Train model
        model.fit(self.X_small, self.y_small_reg)
        assert model.is_trained
        
        # Make predictions
        result = model.predict(self.X_small[:5])
        assert isinstance(result, PredictionResult)
        assert result.predictions.shape == (5,)
        assert result.probabilities is None
    
    @pytest.mark.skipif(
        not pytest.importorskip("sklearn", reason="scikit-learn not available"),
        reason="scikit-learn not available"
    )
    def test_svm_classifier_creation(self):
        """Test SVM classifier creation."""
        model = create_svm_classifier()
        
        assert isinstance(model, BaseModel)
        assert TaskType.CLASSIFICATION in model.capabilities.supported_tasks
        assert model.capabilities.supports_probability_prediction
    
    @pytest.mark.skipif(
        not pytest.importorskip("sklearn", reason="scikit-learn not available"),
        reason="scikit-learn not available"
    )
    def test_gradient_boosting_models(self):
        """Test Gradient Boosting models."""
        # Test classifier
        clf_model = create_gradient_boosting_classifier(n_estimators=10)
        assert isinstance(clf_model, BaseModel)
        assert TaskType.CLASSIFICATION in clf_model.capabilities.supported_tasks
        
        # Test regressor
        reg_model = create_gradient_boosting_regressor(n_estimators=10)
        assert isinstance(reg_model, BaseModel)
        assert TaskType.REGRESSION in reg_model.capabilities.supported_tasks
    
    @pytest.mark.skipif(
        not pytest.importorskip("sklearn", reason="scikit-learn not available"),
        reason="scikit-learn not available"
    )
    def test_sklearn_model_save_load(self):
        """Test saving and loading sklearn models."""
        import tempfile
        import os
        
        model = create_random_forest_classifier(n_estimators=10)
        model.fit(self.X_small, self.y_small_class)
        
        # Test save and load
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "test_model.pkl")
            
            # Save model
            model.save(model_path)
            assert os.path.exists(model_path)
            
            # Load model
            new_model = create_random_forest_classifier()
            new_model.load(model_path)
            
            assert new_model.is_trained
            
            # Test predictions are similar
            original_pred = model.predict(self.X_small[:5])
            loaded_pred = new_model.predict(self.X_small[:5])
            
            np.testing.assert_array_equal(
                original_pred.predictions, 
                loaded_pred.predictions
            )
    
    def test_sklearn_registration(self):
        """Test sklearn model registration."""
        # Clear registry
        registry = get_model_registry()
        registry.clear()
        
        # Register sklearn models
        register_sklearn_models()
        
        # Check that models are registered
        models = registry.list_models()
        expected_models = [
            "random_forest", "logistic_regression", "svm", "gradient_boosting",
            "random_forest_regressor", "linear_regression", "gradient_boosting_regressor"
        ]
        
        for model_name in expected_models:
            assert model_name in models, f"Model {model_name} not registered"
    
    def test_sklearn_registration_without_sklearn(self):
        """Test sklearn registration when sklearn is not available."""
        # Clear registry first
        registry = get_model_registry()
        registry.clear()
        
        with patch('neurolite.core.utils.safe_import', return_value=None):
            # Should not raise exception, just log warning
            register_sklearn_models()


class TestXGBoostModels:
    """Test XGBoost model implementations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.X_small = np.random.rand(20, 4)
        self.y_small_class = np.random.randint(0, 2, 20)
        self.y_small_reg = np.random.rand(20) * 10
    
    @pytest.mark.skipif(
        not pytest.importorskip("xgboost", reason="XGBoost not available"),
        reason="XGBoost not available"
    )
    def test_xgboost_classifier_creation(self):
        """Test XGBoost classifier creation."""
        model = create_xgboost_classifier(n_estimators=10)
        
        assert isinstance(model, BaseModel)
        assert TaskType.CLASSIFICATION in model.capabilities.supported_tasks
        assert DataType.TABULAR in model.capabilities.supported_data_types
        assert model.capabilities.framework == "sklearn"  # Uses sklearn adapter
        assert model.capabilities.supports_feature_importance
        assert model.capabilities.supports_probability_prediction
    
    @pytest.mark.skipif(
        not pytest.importorskip("xgboost", reason="XGBoost not available"),
        reason="XGBoost not available"
    )
    def test_xgboost_classifier_training(self):
        """Test XGBoost classifier training and prediction."""
        model = create_xgboost_classifier(n_estimators=10)
        
        # Train model
        model.fit(self.X_small, self.y_small_class)
        assert model.is_trained
        
        # Make predictions
        result = model.predict(self.X_small[:5])
        assert isinstance(result, PredictionResult)
        assert result.predictions.shape == (5,)
        assert result.probabilities is not None
        assert result.feature_importance is not None
    
    @pytest.mark.skipif(
        not pytest.importorskip("xgboost", reason="XGBoost not available"),
        reason="XGBoost not available"
    )
    def test_xgboost_regressor_creation(self):
        """Test XGBoost regressor creation."""
        model = create_xgboost_regressor(n_estimators=10)
        
        assert isinstance(model, BaseModel)
        assert TaskType.REGRESSION in model.capabilities.supported_tasks
        assert model.capabilities.supports_feature_importance
    
    @pytest.mark.skipif(
        not pytest.importorskip("xgboost", reason="XGBoost not available"),
        reason="XGBoost not available"
    )
    def test_xgboost_regressor_training(self):
        """Test XGBoost regressor training and prediction."""
        model = create_xgboost_regressor(n_estimators=10)
        
        # Train model
        model.fit(self.X_small, self.y_small_reg)
        assert model.is_trained
        
        # Make predictions
        result = model.predict(self.X_small[:5])
        assert isinstance(result, PredictionResult)
        assert result.predictions.shape == (5,)
        assert result.probabilities is None
        assert result.feature_importance is not None
    
    def test_xgboost_registration(self):
        """Test XGBoost model registration."""
        # Clear registry
        registry = get_model_registry()
        registry.clear()
        
        # Register XGBoost models
        register_xgboost_models()
        
        # Check that models are registered
        models = registry.list_models()
        expected_models = ["xgboost", "xgboost_regressor"]
        
        for model_name in expected_models:
            assert model_name in models, f"Model {model_name} not registered"
    
    def test_xgboost_registration_without_xgboost(self):
        """Test XGBoost registration when XGBoost is not available."""
        # Clear registry first
        registry = get_model_registry()
        registry.clear()
        
        with patch('neurolite.core.utils.safe_import', return_value=None):
            # Should not raise exception, just log warning
            register_xgboost_models()


class TestEnsembleModels:
    """Test ensemble model implementations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.X_small = np.random.rand(20, 4)
        self.y_small_class = np.random.randint(0, 2, 20)
        self.y_small_reg = np.random.rand(20) * 10
        
        # Create mock models for testing
        self.mock_models = self._create_mock_models()
    
    def _create_mock_models(self):
        """Create mock models for ensemble testing."""
        models = []
        for i in range(3):
            model = Mock(spec=BaseModel)
            model.is_trained = False
            model.capabilities = ModelCapabilities(
                supported_tasks=[TaskType.CLASSIFICATION],
                supported_data_types=[DataType.TABULAR],
                framework="mock",
                requires_gpu=False,
                min_samples=1,
                supports_probability_prediction=True,
                supports_feature_importance=True
            )
            
            # Create a closure to capture the model instance
            def create_fit_function(m):
                def mock_fit(X, y, validation_data=None, **kwargs):
                    m.is_trained = True
                    return m
                return mock_fit
            
            def mock_predict(X, **kwargs):
                n_samples = len(X)
                predictions = np.random.randint(0, 2, n_samples)
                probabilities = np.random.rand(n_samples, 2)
                probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
                return PredictionResult(
                    predictions=predictions,
                    probabilities=probabilities
                )
            
            def mock_predict_proba(X, **kwargs):
                n_samples = len(X)
                probabilities = np.random.rand(n_samples, 2)
                return probabilities / probabilities.sum(axis=1, keepdims=True)
            
            def mock_get_feature_importance():
                return np.random.rand(4)  # 4 features
            
            model.fit = create_fit_function(model)
            model.predict = mock_predict
            model.predict_proba = mock_predict_proba
            model.get_feature_importance = mock_get_feature_importance
            
            models.append(model)
        
        return models
    
    def test_voting_ensemble_creation(self):
        """Test VotingEnsemble creation."""
        ensemble = VotingEnsemble(models=self.mock_models, voting="soft")
        
        assert isinstance(ensemble, BaseModel)
        assert len(ensemble.models) == 3
        assert ensemble.voting == "soft"
        assert ensemble.weights is None
    
    def test_voting_ensemble_capabilities(self):
        """Test VotingEnsemble capabilities."""
        ensemble = VotingEnsemble(models=self.mock_models)
        capabilities = ensemble.capabilities
        
        assert TaskType.CLASSIFICATION in capabilities.supported_tasks
        assert DataType.TABULAR in capabilities.supported_data_types
        assert capabilities.framework == "ensemble"
        assert capabilities.supports_probability_prediction
        assert capabilities.supports_feature_importance
    
    def test_voting_ensemble_training(self):
        """Test VotingEnsemble training."""
        ensemble = VotingEnsemble(models=self.mock_models)
        
        # Train ensemble
        ensemble.fit(self.X_small, self.y_small_class)
        assert ensemble.is_trained
        
        # Check that all base models were trained
        for model in ensemble.models:
            assert model.is_trained
    
    def test_voting_ensemble_prediction_soft(self):
        """Test VotingEnsemble soft voting prediction."""
        ensemble = VotingEnsemble(models=self.mock_models, voting="soft")
        ensemble.fit(self.X_small, self.y_small_class)
        
        # Make predictions
        result = ensemble.predict(self.X_small[:5])
        assert isinstance(result, PredictionResult)
        assert result.predictions.shape == (5,)
        assert result.probabilities is not None
        assert result.probabilities.shape == (5, 2)
        assert result.feature_importance is not None
    
    def test_voting_ensemble_prediction_hard(self):
        """Test VotingEnsemble hard voting prediction."""
        ensemble = VotingEnsemble(models=self.mock_models, voting="hard")
        ensemble.fit(self.X_small, self.y_small_class)
        
        # Make predictions
        result = ensemble.predict(self.X_small[:5])
        assert isinstance(result, PredictionResult)
        assert result.predictions.shape == (5,)
        # Hard voting might not have probabilities
    
    def test_voting_ensemble_weighted(self):
        """Test VotingEnsemble with weights."""
        weights = [0.5, 0.3, 0.2]
        ensemble = VotingEnsemble(models=self.mock_models, weights=weights)
        
        assert ensemble.weights == weights
        
        # Test training and prediction
        ensemble.fit(self.X_small, self.y_small_class)
        result = ensemble.predict(self.X_small[:5])
        assert isinstance(result, PredictionResult)
    
    def test_voting_ensemble_invalid_weights(self):
        """Test VotingEnsemble with invalid weights."""
        with pytest.raises(ValueError, match="Number of weights must match number of models"):
            VotingEnsemble(models=self.mock_models, weights=[0.5, 0.5])  # Wrong number of weights
    
    def test_stacking_ensemble_creation(self):
        """Test StackingEnsemble creation."""
        meta_model = Mock(spec=BaseModel)
        meta_model.capabilities = ModelCapabilities(
            supported_tasks=[TaskType.CLASSIFICATION],
            supported_data_types=[DataType.TABULAR],
            framework="mock",
            requires_gpu=False,
            min_samples=1,
            supports_probability_prediction=True,
            supports_feature_importance=True
        )
        
        ensemble = StackingEnsemble(
            base_models=self.mock_models,
            meta_model=meta_model,
            cv_folds=3
        )
        
        assert isinstance(ensemble, BaseModel)
        assert len(ensemble.base_models) == 3
        assert ensemble.meta_model == meta_model
        assert ensemble.cv_folds == 3
    
    def test_stacking_ensemble_capabilities(self):
        """Test StackingEnsemble capabilities."""
        meta_model = Mock(spec=BaseModel)
        meta_model.capabilities = ModelCapabilities(
            supported_tasks=[TaskType.CLASSIFICATION],
            supported_data_types=[DataType.TABULAR],
            framework="mock",
            requires_gpu=False,
            min_samples=1,
            supports_probability_prediction=True,
            supports_feature_importance=True
        )
        
        ensemble = StackingEnsemble(
            base_models=self.mock_models,
            meta_model=meta_model
        )
        
        capabilities = ensemble.capabilities
        assert TaskType.CLASSIFICATION in capabilities.supported_tasks
        assert DataType.TABULAR in capabilities.supported_data_types
        assert capabilities.framework == "ensemble"
    
    @pytest.mark.skipif(
        not pytest.importorskip("sklearn", reason="scikit-learn not available"),
        reason="scikit-learn not available"
    )
    def test_ensemble_factory_functions(self):
        """Test ensemble factory functions."""
        # Test with explicit models to avoid dependency issues
        mock_models = self.mock_models
        
        # Test voting classifier creation with explicit models
        voting_clf = create_voting_classifier(models=mock_models)
        assert isinstance(voting_clf, VotingEnsemble)
        
        # Test voting regressor creation with explicit models
        voting_reg = create_voting_regressor(models=mock_models)
        assert isinstance(voting_reg, VotingEnsemble)
        
        # Create mock meta model
        meta_model = Mock(spec=BaseModel)
        meta_model.capabilities = ModelCapabilities(
            supported_tasks=[TaskType.CLASSIFICATION],
            supported_data_types=[DataType.TABULAR],
            framework="mock",
            requires_gpu=False,
            min_samples=1,
            supports_probability_prediction=True,
            supports_feature_importance=True
        )
        
        # Test stacking classifier creation with explicit models
        stacking_clf = create_stacking_classifier(base_models=mock_models, meta_model=meta_model)
        assert isinstance(stacking_clf, StackingEnsemble)
        
        # Test stacking regressor creation with explicit models
        stacking_reg = create_stacking_regressor(base_models=mock_models, meta_model=meta_model)
        assert isinstance(stacking_reg, StackingEnsemble)
    
    def test_ensemble_registration(self):
        """Test ensemble model registration."""
        # Clear registry
        registry = get_model_registry()
        registry.clear()
        
        # First register some base models that ensemble models depend on
        register_sklearn_models()
        
        # Register ensemble models
        register_ensemble_models()
        
        # Check that models are registered
        models = registry.list_models()
        expected_models = [
            "voting_classifier", "voting_regressor",
            "stacking_classifier", "stacking_regressor"
        ]
        
        for model_name in expected_models:
            assert model_name in models, f"Model {model_name} not registered"
    
    def test_ensemble_registration_without_sklearn(self):
        """Test ensemble registration when sklearn is not available."""
        # Clear registry first
        registry = get_model_registry()
        registry.clear()
        
        with patch('neurolite.core.utils.safe_import', return_value=None):
            # Should not raise exception, just log warning
            register_ensemble_models()


class TestMLModelIntegration:
    """Integration tests for ML models."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.X = np.random.rand(50, 4)
        self.y_class = np.random.randint(0, 2, 50)
        self.y_reg = np.random.rand(50) * 10
    
    @pytest.mark.skipif(
        not pytest.importorskip("sklearn", reason="scikit-learn not available"),
        reason="scikit-learn not available"
    )
    def test_full_ml_workflow_classification(self):
        """Test complete ML workflow for classification."""
        from neurolite.models.factory import create_model
        from neurolite.models import initialize_models
        
        # Initialize models first
        initialize_models()
        
        # Create and train model
        model = create_model("random_forest", n_estimators=10)
        model.fit(self.X, self.y_class)
        
        # Make predictions
        result = model.predict(self.X[:10])
        assert result.predictions.shape == (10,)
        assert result.probabilities is not None
        assert result.feature_importance is not None
        
        # Test probability prediction
        probabilities = model.predict_proba(self.X[:10])
        assert probabilities.shape == (10, 2)
    
    @pytest.mark.skipif(
        not pytest.importorskip("sklearn", reason="scikit-learn not available"),
        reason="scikit-learn not available"
    )
    def test_full_ml_workflow_regression(self):
        """Test complete ML workflow for regression."""
        from neurolite.models.factory import create_model
        from neurolite.models import initialize_models
        
        # Initialize models first
        initialize_models()
        
        # Create and train model
        model = create_model("random_forest_regressor", n_estimators=10)
        model.fit(self.X, self.y_reg)
        
        # Make predictions
        result = model.predict(self.X[:10])
        assert result.predictions.shape == (10,)
        assert result.probabilities is None  # Regression doesn't have probabilities
        assert result.feature_importance is not None
    
    @pytest.mark.skipif(
        not pytest.importorskip("xgboost", reason="XGBoost not available"),
        reason="XGBoost not available"
    )
    def test_xgboost_integration(self):
        """Test XGBoost integration."""
        from neurolite.models.factory import create_model
        from neurolite.models import initialize_models
        
        # Initialize models first
        initialize_models()
        
        # Test classifier
        clf_model = create_model("xgboost", n_estimators=10)
        clf_model.fit(self.X, self.y_class)
        clf_result = clf_model.predict(self.X[:10])
        assert clf_result.predictions.shape == (10,)
        
        # Test regressor
        reg_model = create_model("xgboost_regressor", n_estimators=10)
        reg_model.fit(self.X, self.y_reg)
        reg_result = reg_model.predict(self.X[:10])
        assert reg_result.predictions.shape == (10,)
    
    def test_model_registry_integration(self):
        """Test that all ML models are properly registered."""
        from neurolite.models import initialize_models
        from neurolite.models.registry import get_model_registry
        
        # Initialize models
        initialize_models()
        
        # Get registry
        registry = get_model_registry()
        models = registry.list_models()
        
        # Check that we have models registered
        assert len(models) > 0
        
        # Check for specific model types
        classification_models = registry.list_models(task_type=TaskType.CLASSIFICATION)
        regression_models = registry.list_models(task_type=TaskType.REGRESSION)
        
        assert len(classification_models) > 0
        assert len(regression_models) > 0
    
    def test_auto_model_selection(self):
        """Test automatic model selection for ML tasks."""
        from neurolite.models.registry import auto_select_model
        from neurolite.models import initialize_models
        
        # Initialize models
        initialize_models()
        
        # Test auto-selection for classification
        selected_clf = auto_select_model(
            task_type=TaskType.CLASSIFICATION,
            data_type=DataType.TABULAR,
            num_samples=100
        )
        assert selected_clf is not None
        
        # Test auto-selection for regression
        selected_reg = auto_select_model(
            task_type=TaskType.REGRESSION,
            data_type=DataType.TABULAR,
            num_samples=100
        )
        assert selected_reg is not None


if __name__ == "__main__":
    pytest.main([__file__])