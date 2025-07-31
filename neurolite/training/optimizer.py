"""
Hyperparameter optimization integration using Optuna.

Provides automated hyperparameter search across different model types
with resource-aware termination and progress monitoring.
"""

import time
import warnings
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

try:
    import optuna
    from optuna import Trial
    from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler
    from optuna.pruners import MedianPruner, HyperbandPruner, SuccessiveHalvingPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None
    Trial = None

from ..models.base import BaseModel, TaskType
from ..data.detector import DataType
from ..core import get_logger
from ..core.exceptions import TrainingError, ConfigurationError
from .config import TrainingConfiguration, OptimizerType, LossFunction, SchedulerType
from .trainer import TrainingEngine, TrainedModel

logger = get_logger(__name__)


class SearchStrategy(Enum):
    """Hyperparameter search strategies."""
    TPE = "tpe"  # Tree-structured Parzen Estimator
    RANDOM = "random"
    CMAES = "cmaes"  # Covariance Matrix Adaptation Evolution Strategy


class PruningStrategy(Enum):
    """Early pruning strategies for optimization."""
    MEDIAN = "median"
    HYPERBAND = "hyperband"
    SUCCESSIVE_HALVING = "successive_halving"
    NONE = "none"


@dataclass
class OptimizationBounds:
    """Bounds for hyperparameter optimization."""
    learning_rate: Tuple[float, float] = (1e-5, 1e-1)
    batch_size: List[int] = field(default_factory=lambda: [8, 16, 32, 64, 128])
    epochs: Tuple[int, int] = (10, 200)
    dropout_rate: Tuple[float, float] = (0.0, 0.5)
    l1_regularization: Tuple[float, float] = (0.0, 0.1)
    l2_regularization: Tuple[float, float] = (0.0, 0.1)
    weight_decay: Tuple[float, float] = (0.0, 0.01)
    
    # Optimizer-specific bounds
    momentum: Tuple[float, float] = (0.8, 0.99)
    beta1: Tuple[float, float] = (0.8, 0.99)
    beta2: Tuple[float, float] = (0.9, 0.999)
    
    # Scheduler-specific bounds
    scheduler_step_size: Tuple[int, int] = (10, 50)
    scheduler_gamma: Tuple[float, float] = (0.1, 0.9)
    scheduler_patience: Tuple[int, int] = (5, 20)


@dataclass
class ResourceConstraints:
    """Resource constraints for optimization."""
    max_trials: int = 100
    max_time_minutes: Optional[int] = None
    max_memory_gb: Optional[float] = None
    max_gpu_memory_gb: Optional[float] = None
    early_stopping_rounds: int = 10
    min_trials_before_pruning: int = 5


@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization."""
    search_strategy: SearchStrategy = SearchStrategy.TPE
    pruning_strategy: PruningStrategy = PruningStrategy.MEDIAN
    optimization_bounds: OptimizationBounds = field(default_factory=OptimizationBounds)
    resource_constraints: ResourceConstraints = field(default_factory=ResourceConstraints)
    
    # Optimization target
    objective_metric: str = "val_loss"
    objective_direction: str = "minimize"  # "minimize" or "maximize"
    
    # Cross-validation settings
    cv_folds: int = 3
    use_cross_validation: bool = True
    
    # Parallel execution
    n_jobs: int = 1
    
    # Logging and monitoring
    verbose: bool = True
    save_study: bool = True
    study_name: Optional[str] = None
    storage_url: Optional[str] = None  # For distributed optimization
    
    # Custom parameter spaces
    custom_parameter_space: Optional[Dict[str, Any]] = None


@dataclass
class OptimizationResult:
    """Result of hyperparameter optimization."""
    best_params: Dict[str, Any]
    best_value: float
    best_trial: Optional[Any] = None
    study: Optional[Any] = None
    optimization_time: float = 0.0
    n_trials: int = 0
    best_model: Optional[TrainedModel] = None
    trial_history: List[Dict[str, Any]] = field(default_factory=list)


class HyperparameterOptimizer:
    """
    Hyperparameter optimizer using Optuna for automated search.
    
    Integrates with the training engine to find optimal hyperparameters
    for different model types with resource-aware termination.
    """
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        """
        Initialize the hyperparameter optimizer.
        
        Args:
            config: Optimization configuration
            
        Raises:
            ImportError: If Optuna is not available
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna is required for hyperparameter optimization. "
                "Install it with: pip install optuna"
            )
        
        self.config = config or OptimizationConfig()
        self.study: Optional[optuna.Study] = None
        self._start_time: Optional[float] = None
        self._best_model: Optional[TrainedModel] = None
        
        # Suppress Optuna's verbose logging if needed
        if not self.config.verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        logger.debug("Initialized HyperparameterOptimizer")
    
    def optimize(
        self,
        model_factory: Callable[..., BaseModel],
        X_train: Any,
        y_train: Any,
        X_val: Optional[Any] = None,
        y_val: Optional[Any] = None,
        task_type: Optional[TaskType] = None,
        data_type: Optional[DataType] = None,
        base_config: Optional[TrainingConfiguration] = None,
        **kwargs
    ) -> OptimizationResult:
        """
        Optimize hyperparameters for the given model and data.
        
        Args:
            model_factory: Function that creates model instances
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            task_type: Type of ML task
            data_type: Type of input data
            base_config: Base training configuration
            **kwargs: Additional parameters
            
        Returns:
            OptimizationResult containing best parameters and model
        """
        logger.info("Starting hyperparameter optimization")
        self._start_time = time.time()
        
        # Create study
        self.study = self._create_study()
        
        # Create objective function
        objective_func = self._create_objective_function(
            model_factory=model_factory,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            task_type=task_type,
            data_type=data_type,
            base_config=base_config,
            **kwargs
        )
        
        # Run optimization
        try:
            self.study.optimize(
                objective_func,
                n_trials=self.config.resource_constraints.max_trials,
                timeout=self._get_timeout_seconds(),
                n_jobs=self.config.n_jobs,
                callbacks=[self._create_callback()]
            )
        except KeyboardInterrupt:
            logger.info("Optimization interrupted by user")
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise TrainingError(f"Hyperparameter optimization failed: {e}") from e
        
        # Create result
        optimization_time = time.time() - self._start_time
        result = OptimizationResult(
            best_params=self.study.best_params,
            best_value=self.study.best_value,
            best_trial=self.study.best_trial,
            study=self.study,
            optimization_time=optimization_time,
            n_trials=len(self.study.trials),
            best_model=self._best_model,
            trial_history=self._get_trial_history()
        )
        
        logger.info(f"Optimization completed in {optimization_time:.2f} seconds")
        logger.info(f"Best value: {result.best_value:.6f}")
        logger.info(f"Best parameters: {result.best_params}")
        
        return result
    
    def _create_study(self) -> optuna.Study:
        """Create Optuna study with configured sampler and pruner."""
        # Create sampler
        if self.config.search_strategy == SearchStrategy.TPE:
            sampler = TPESampler()
        elif self.config.search_strategy == SearchStrategy.RANDOM:
            sampler = RandomSampler()
        elif self.config.search_strategy == SearchStrategy.CMAES:
            sampler = CmaEsSampler()
        else:
            sampler = TPESampler()  # Default
        
        # Create pruner
        if self.config.pruning_strategy == PruningStrategy.MEDIAN:
            pruner = MedianPruner(
                n_startup_trials=self.config.resource_constraints.min_trials_before_pruning,
                n_warmup_steps=5
            )
        elif self.config.pruning_strategy == PruningStrategy.HYPERBAND:
            pruner = HyperbandPruner(
                min_resource=1,
                max_resource=100,
                reduction_factor=3
            )
        elif self.config.pruning_strategy == PruningStrategy.SUCCESSIVE_HALVING:
            pruner = SuccessiveHalvingPruner()
        else:
            pruner = MedianPruner()  # Default
        
        # Determine direction
        direction = "minimize" if self.config.objective_direction == "minimize" else "maximize"
        
        # Create study
        study_name = self.config.study_name or f"optimization_{int(time.time())}"
        
        study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            storage=self.config.storage_url
        )
        
        return study
    
    def _create_objective_function(
        self,
        model_factory: Callable[..., BaseModel],
        X_train: Any,
        y_train: Any,
        X_val: Optional[Any] = None,
        y_val: Optional[Any] = None,
        task_type: Optional[TaskType] = None,
        data_type: Optional[DataType] = None,
        base_config: Optional[TrainingConfiguration] = None,
        **kwargs
    ) -> Callable[[Trial], float]:
        """Create objective function for Optuna optimization."""
        
        def objective(trial: Trial) -> float:
            try:
                # Sample hyperparameters
                params = self._sample_hyperparameters(trial, task_type, data_type)
                
                # Create model
                model = model_factory(**params.get('model_params', {}))
                
                # Create training configuration
                training_config = self._create_training_config(params, base_config)
                
                # Create training engine
                trainer = TrainingEngine(training_config)
                
                # Train model
                if self.config.use_cross_validation and X_val is None:
                    # Use cross-validation
                    score = self._cross_validate(
                        model, trainer, X_train, y_train, trial
                    )
                else:
                    # Use validation split
                    trained_model = trainer.train(
                        model=model,
                        X_train=X_train,
                        y_train=y_train,
                        X_val=X_val,
                        y_val=y_val,
                        task_type=task_type,
                        data_type=data_type
                    )
                    
                    # Get objective value
                    score = self._get_objective_value(trained_model, trial)
                    
                    # Store best model
                    if (self._best_model is None or 
                        (self.config.objective_direction == "minimize" and score < self.study.best_value) or
                        (self.config.objective_direction == "maximize" and score > self.study.best_value)):
                        self._best_model = trained_model
                
                return score
                
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                # Return worst possible value for failed trials
                return float('inf') if self.config.objective_direction == "minimize" else float('-inf')
        
        return objective
    
    def _sample_hyperparameters(
        self, 
        trial: Trial, 
        task_type: Optional[TaskType] = None,
        data_type: Optional[DataType] = None
    ) -> Dict[str, Any]:
        """Sample hyperparameters for the trial."""
        params = {}
        bounds = self.config.optimization_bounds
        
        # Learning rate
        params['learning_rate'] = trial.suggest_float(
            'learning_rate',
            bounds.learning_rate[0],
            bounds.learning_rate[1],
            log=True
        )
        
        # Batch size
        params['batch_size'] = trial.suggest_categorical(
            'batch_size',
            bounds.batch_size
        )
        
        # Epochs
        params['epochs'] = trial.suggest_int(
            'epochs',
            bounds.epochs[0],
            bounds.epochs[1]
        )
        
        # Regularization
        params['dropout_rate'] = trial.suggest_float(
            'dropout_rate',
            bounds.dropout_rate[0],
            bounds.dropout_rate[1]
        )
        
        params['l1_regularization'] = trial.suggest_float(
            'l1_regularization',
            bounds.l1_regularization[0],
            bounds.l1_regularization[1]
        )
        
        params['l2_regularization'] = trial.suggest_float(
            'l2_regularization',
            bounds.l2_regularization[0],
            bounds.l2_regularization[1]
        )
        
        # Optimizer type
        optimizer_type = trial.suggest_categorical(
            'optimizer_type',
            [opt.value for opt in OptimizerType]
        )
        params['optimizer_type'] = OptimizerType(optimizer_type)
        
        # Optimizer-specific parameters
        if params['optimizer_type'] in [OptimizerType.ADAM, OptimizerType.ADAMW]:
            params['beta1'] = trial.suggest_float(
                'beta1',
                bounds.beta1[0],
                bounds.beta1[1]
            )
            params['beta2'] = trial.suggest_float(
                'beta2',
                bounds.beta2[0],
                bounds.beta2[1]
            )
        elif params['optimizer_type'] == OptimizerType.SGD:
            params['momentum'] = trial.suggest_float(
                'momentum',
                bounds.momentum[0],
                bounds.momentum[1]
            )
        
        params['weight_decay'] = trial.suggest_float(
            'weight_decay',
            bounds.weight_decay[0],
            bounds.weight_decay[1]
        )
        
        # Learning rate scheduler
        scheduler_type = trial.suggest_categorical(
            'scheduler_type',
            [sched.value for sched in SchedulerType]
        )
        params['scheduler_type'] = SchedulerType(scheduler_type)
        
        if params['scheduler_type'] == SchedulerType.STEP:
            params['scheduler_step_size'] = trial.suggest_int(
                'scheduler_step_size',
                bounds.scheduler_step_size[0],
                bounds.scheduler_step_size[1]
            )
            params['scheduler_gamma'] = trial.suggest_float(
                'scheduler_gamma',
                bounds.scheduler_gamma[0],
                bounds.scheduler_gamma[1]
            )
        elif params['scheduler_type'] == SchedulerType.REDUCE_ON_PLATEAU:
            params['scheduler_patience'] = trial.suggest_int(
                'scheduler_patience',
                bounds.scheduler_patience[0],
                bounds.scheduler_patience[1]
            )
        
        # Add custom parameter space if provided
        if self.config.custom_parameter_space:
            for param_name, param_config in self.config.custom_parameter_space.items():
                if param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config['low'],
                        param_config['high'],
                        log=param_config.get('log', False)
                    )
                elif param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config['low'],
                        param_config['high']
                    )
                elif param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_config['choices']
                    )
        
        return params
    
    def _create_training_config(
        self, 
        params: Dict[str, Any], 
        base_config: Optional[TrainingConfiguration] = None
    ) -> TrainingConfiguration:
        """Create training configuration from sampled parameters."""
        config = base_config or TrainingConfiguration()
        
        # Update basic parameters
        config.batch_size = params.get('batch_size', config.batch_size)
        config.epochs = params.get('epochs', config.epochs)
        config.dropout_rate = params.get('dropout_rate', config.dropout_rate)
        config.l1_regularization = params.get('l1_regularization', config.l1_regularization)
        config.l2_regularization = params.get('l2_regularization', config.l2_regularization)
        
        # Update optimizer configuration
        config.optimizer.type = params.get('optimizer_type', config.optimizer.type)
        config.optimizer.learning_rate = params.get('learning_rate', config.optimizer.learning_rate)
        config.optimizer.weight_decay = params.get('weight_decay', config.optimizer.weight_decay)
        config.optimizer.momentum = params.get('momentum', config.optimizer.momentum)
        config.optimizer.beta1 = params.get('beta1', config.optimizer.beta1)
        config.optimizer.beta2 = params.get('beta2', config.optimizer.beta2)
        
        # Update scheduler configuration
        config.scheduler.type = params.get('scheduler_type', config.scheduler.type)
        config.scheduler.step_size = params.get('scheduler_step_size', config.scheduler.step_size)
        config.scheduler.gamma = params.get('scheduler_gamma', config.scheduler.gamma)
        config.scheduler.patience = params.get('scheduler_patience', config.scheduler.patience)
        
        return config
    
    def _cross_validate(
        self,
        model: BaseModel,
        trainer: TrainingEngine,
        X: Any,
        y: Any,
        trial: Trial
    ) -> float:
        """Perform cross-validation and return average score."""
        from sklearn.model_selection import KFold, StratifiedKFold
        
        # Determine CV strategy
        if hasattr(y, 'nunique') and y.nunique() < len(y) * 0.1:
            # Classification task - use stratified CV
            cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=42)
        else:
            # Regression task - use regular CV
            cv = KFold(n_splits=self.config.cv_folds, shuffle=True, random_state=42)
        
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            # Split data
            if hasattr(X, 'iloc'):  # pandas DataFrame
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            else:  # numpy array
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Create fresh model instance
            model_class = model.__class__
            fold_model = model_class(**model.get_config())
            
            # Train model
            trained_model = trainer.train(
                model=fold_model,
                X_train=X_train_fold,
                y_train=y_train_fold,
                X_val=X_val_fold,
                y_val=y_val_fold
            )
            
            # Get score
            score = self._get_objective_value(trained_model, trial, fold)
            scores.append(score)
            
            # Report intermediate value for pruning
            trial.report(score, fold)
            
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return np.mean(scores)
    
    def _get_objective_value(
        self, 
        trained_model: TrainedModel, 
        trial: Trial,
        fold: Optional[int] = None
    ) -> float:
        """Extract objective value from trained model."""
        metric_name = self.config.objective_metric
        
        # Try to get the metric from training history
        if metric_name in trained_model.history.metrics:
            values = trained_model.history.metrics[metric_name]
            if values:
                return float(values[-1])  # Last epoch value
        
        # Try validation loss
        if metric_name == "val_loss" and trained_model.history.val_loss:
            return float(trained_model.history.val_loss[-1])
        
        # Try training loss
        if metric_name == "train_loss" and trained_model.history.train_loss:
            return float(trained_model.history.train_loss[-1])
        
        # Default to validation loss or training loss
        if trained_model.history.val_loss:
            return float(trained_model.history.val_loss[-1])
        elif trained_model.history.train_loss:
            return float(trained_model.history.train_loss[-1])
        
        # If no metrics available, return worst possible value
        logger.warning(f"Could not find metric '{metric_name}' in training history")
        return float('inf') if self.config.objective_direction == "minimize" else float('-inf')
    
    def _get_timeout_seconds(self) -> Optional[int]:
        """Get timeout in seconds from configuration."""
        if self.config.resource_constraints.max_time_minutes:
            return self.config.resource_constraints.max_time_minutes * 60
        return None
    
    def _create_callback(self) -> Callable[[optuna.Study, optuna.Trial], None]:
        """Create callback for monitoring optimization progress."""
        
        def callback(study: optuna.Study, trial: optuna.Trial) -> None:
            if self.config.verbose:
                logger.info(
                    f"Trial {trial.number} finished with value: {trial.value:.6f} "
                    f"and parameters: {trial.params}"
                )
            
            # Check resource constraints
            if self._start_time:
                elapsed_time = time.time() - self._start_time
                if (self.config.resource_constraints.max_time_minutes and 
                    elapsed_time > self.config.resource_constraints.max_time_minutes * 60):
                    study.stop()
                    logger.info("Stopping optimization due to time limit")
        
        return callback
    
    def _get_trial_history(self) -> List[Dict[str, Any]]:
        """Get history of all trials."""
        if not self.study:
            return []
        
        history = []
        for trial in self.study.trials:
            history.append({
                'number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': trial.state.name,
                'datetime_start': trial.datetime_start,
                'datetime_complete': trial.datetime_complete,
                'duration': trial.duration
            })
        
        return history
    
    def get_optimization_insights(self) -> Dict[str, Any]:
        """Get insights from the optimization process."""
        if not self.study:
            return {}
        
        insights = {
            'n_trials': len(self.study.trials),
            'best_value': self.study.best_value,
            'best_params': self.study.best_params,
            'optimization_time': time.time() - self._start_time if self._start_time else 0
        }
        
        # Parameter importance
        try:
            importance = optuna.importance.get_param_importances(self.study)
            insights['param_importance'] = importance
        except Exception as e:
            logger.debug(f"Could not compute parameter importance: {e}")
        
        # Optimization history
        insights['optimization_history'] = [
            {'trial': i, 'value': trial.value}
            for i, trial in enumerate(self.study.trials)
            if trial.value is not None
        ]
        
        return insights


def optimize_hyperparameters(
    model_factory: Callable[..., BaseModel],
    X_train: Any,
    y_train: Any,
    X_val: Optional[Any] = None,
    y_val: Optional[Any] = None,
    task_type: Optional[TaskType] = None,
    data_type: Optional[DataType] = None,
    config: Optional[OptimizationConfig] = None,
    **kwargs
) -> OptimizationResult:
    """
    Convenience function for hyperparameter optimization.
    
    Args:
        model_factory: Function that creates model instances
        X_train: Training features
        y_train: Training targets
        X_val: Validation features (optional)
        y_val: Validation targets (optional)
        task_type: Type of ML task
        data_type: Type of input data
        config: Optimization configuration
        **kwargs: Additional parameters
        
    Returns:
        OptimizationResult containing best parameters and model
    """
    optimizer = HyperparameterOptimizer(config)
    return optimizer.optimize(
        model_factory=model_factory,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        task_type=task_type,
        data_type=data_type,
        **kwargs
    )