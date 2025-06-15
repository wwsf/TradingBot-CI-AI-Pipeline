import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import ParameterGrid
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import json
import itertools
from datetime import datetime


class AIParameterOptimizer:
    """
    AI-powered parameter optimization for trading strategies.

    This is the brain of our CI/CD pipeline - it automatically
    finds the best parameters for any trading strategy!
    """

    def __init__(self, strategy_class, optimization_metric: str = "sharpe_ratio"):
        """
        Initialize the AI optimizer

        Args:
            strategy_class: The trading strategy class to optimize
            optimization_metric: What metric to optimize ('sharpe_ratio', 'annual_return', etc.)
        """
        self.strategy_class = strategy_class
        self.optimization_metric = optimization_metric
        self.optimization_history = []
        self.best_params = None
        self.best_score = float('-inf')

        print(f"🤖 AI Optimizer initialized for {strategy_class.__name__}")
        print(f"🎯 Optimizing for: {optimization_metric}")

    def grid_search_optimization(self, symbol: str, start_date: str, end_date: str,
                                 max_iterations: int = 50) -> Dict:
        """
        Basic grid search optimization
        Good for initial exploration of parameter space
        """
        print(f"🔍 Starting Grid Search Optimization...")
        print(f"📊 Testing {symbol} from {start_date} to {end_date}")

        # Create a sample strategy to get optimization parameters
        sample_strategy = self.strategy_class()
        param_configs = sample_strategy.get_optimization_params()

        if not param_configs:
            print("❌ No optimization parameters defined for this strategy")
            return {"error": "No optimization parameters"}

        # Generate parameter grid
        param_grid = self._generate_parameter_grid(param_configs, max_iterations)

        print(f"🔢 Testing {len(param_grid)} parameter combinations...")

        results = []

        for i, params in enumerate(param_grid):
            print(f"   Testing {i + 1}/{len(param_grid)}: {params}")

            try:
                # Create strategy with these parameters
                strategy = self.strategy_class(**params)

                # Run backtest
                backtest_result = strategy.backtest(symbol, start_date, end_date)

                if "error" not in backtest_result:
                    score = backtest_result.get(self.optimization_metric, float('-inf'))

                    result = {
                        "parameters": params,
                        "score": score,
                        "metrics": backtest_result
                    }

                    results.append(result)

                    # Track best result
                    if score > self.best_score:
                        self.best_score = score
                        self.best_params = params
                        print(f"   🌟 New best score: {score:.4f}")

                else:
                    print(f"   ❌ Error: {backtest_result['error']}")

            except Exception as e:
                print(f"   ❌ Exception: {str(e)}")
                continue

        # Store optimization history
        self.optimization_history = results

        if self.best_params:
            print(f"✅ Optimization complete!")
            print(f"🏆 Best parameters: {self.best_params}")
            print(f"📊 Best {self.optimization_metric}: {self.best_score:.4f}")

            return {
                "best_parameters": self.best_params,
                "best_score": self.best_score,
                "optimization_metric": self.optimization_metric,
                "total_combinations_tested": len(results),
                "optimization_history": results[-10:]  # Last 10 results
            }
        else:
            return {"error": "No valid parameter combinations found"}

    def bayesian_optimization(self, symbol: str, start_date: str, end_date: str,
                              max_iterations: int = 30) -> Dict:
        """
        Advanced Bayesian optimization using Gaussian Processes
        More efficient than grid search - uses AI to predict best parameters!
        """
        print(f"🧠 Starting Bayesian Optimization (AI-powered)...")
        print(f"📊 Testing {symbol} from {start_date} to {end_date}")

        # Get parameter configurations
        sample_strategy = self.strategy_class()
        param_configs = sample_strategy.get_optimization_params()

        if not param_configs:
            return {"error": "No optimization parameters"}

        # Convert parameters to numerical ranges for Gaussian Process
        param_bounds, param_names = self._prepare_bayesian_params(param_configs)

        if len(param_bounds) == 0:
            return {"error": "No numerical parameters to optimize"}

        # Initialize with a few random samples
        n_initial = min(10, max_iterations // 3)
        X_samples = []
        y_samples = []

        print(f"🎲 Generating {n_initial} initial random samples...")

        for i in range(n_initial):
            # Generate random parameters within bounds
            random_params = {}
            param_values = []

            for param_name, (min_val, max_val) in zip(param_names, param_bounds):
                if param_configs[param_name]["type"] == "integer":
                    value = np.random.randint(min_val, max_val + 1)
                else:
                    value = np.random.uniform(min_val, max_val)

                random_params[param_name] = value
                param_values.append(value)

            # Test these parameters
            score = self._evaluate_parameters(random_params, symbol, start_date, end_date)

            if score is not None:
                X_samples.append(param_values)
                y_samples.append(score)

                if score > self.best_score:
                    self.best_score = score
                    self.best_params = random_params
                    print(f"   🌟 New best: {score:.4f} with {random_params}")

        if len(X_samples) < 3:
            print("❌ Not enough valid samples for Bayesian optimization")
            return {"error": "Insufficient valid samples"}

        # Initialize Gaussian Process
        kernel = Matern(length_scale=1.0, nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)

        print(f"🤖 Starting AI-guided search for {max_iterations - n_initial} more iterations...")

        # Bayesian optimization loop
        for iteration in range(n_initial, max_iterations):
            # Fit Gaussian Process to current data
            X_array = np.array(X_samples)
            y_array = np.array(y_samples)

            gp.fit(X_array, y_array)

            # Find next best point to sample using acquisition function
            next_params, next_values = self._acquisition_function(gp, param_bounds, param_names, param_configs)

            print(f"   🔍 Iteration {iteration + 1}: Testing AI suggestion {next_params}")

            # Evaluate the AI's suggestion
            score = self._evaluate_parameters(next_params, symbol, start_date, end_date)

            if score is not None:
                X_samples.append(next_values)
                y_samples.append(score)

                if score > self.best_score:
                    self.best_score = score
                    self.best_params = next_params
                    print(f"   🌟 New best: {score:.4f}")

        print(f"✅ Bayesian Optimization complete!")
        print(f"🏆 Best parameters: {self.best_params}")
        print(f"📊 Best {self.optimization_metric}: {self.best_score:.4f}")

        return {
            "best_parameters": self.best_params,
            "best_score": self.best_score,
            "optimization_metric": self.optimization_metric,
            "total_evaluations": len(X_samples),
            "optimization_method": "bayesian"
        }

    def _generate_parameter_grid(self, param_configs: Dict, max_iterations: int) -> List[Dict]:
        """Generate parameter grid for grid search"""
        param_ranges = {}

        for param_name, config in param_configs.items():
            if config["type"] == "integer":
                # Create reasonable range of integers
                min_val = config["min"]
                max_val = config["max"]
                step = max(1, (max_val - min_val) // 10)  # About 10 values per parameter
                param_ranges[param_name] = list(range(min_val, max_val + 1, step))

            elif config["type"] == "float":
                # Create range of floats
                min_val = config["min"]
                max_val = config["max"]
                values = np.linspace(min_val, max_val, 10)  # 10 values per parameter
                param_ranges[param_name] = values.tolist()

        # Generate all combinations
        param_grid = list(ParameterGrid(param_ranges))

        # Limit to max_iterations if too many combinations
        if len(param_grid) > max_iterations:
            # Randomly sample from the grid
            indices = np.random.choice(len(param_grid), max_iterations, replace=False)
            param_grid = [param_grid[i] for i in indices]

        return param_grid

    def _prepare_bayesian_params(self, param_configs: Dict) -> Tuple[List[Tuple], List[str]]:
        """Prepare parameters for Bayesian optimization"""
        param_bounds = []
        param_names = []

        for param_name, config in param_configs.items():
            if config["type"] in ["integer", "float"]:
                param_bounds.append((config["min"], config["max"]))
                param_names.append(param_name)

        return param_bounds, param_names

    def _evaluate_parameters(self, params: Dict, symbol: str, start_date: str, end_date: str) -> float:
        """Evaluate a set of parameters and return the score"""
        try:
            strategy = self.strategy_class(**params)
            result = strategy.backtest(symbol, start_date, end_date)

            if "error" not in result:
                score = result.get(self.optimization_metric, float('-inf'))

                # Store in optimization history
                self.optimization_history.append({
                    "parameters": params,
                    "score": score,
                    "metrics": result
                })

                return score
            else:
                return None

        except Exception as e:
            print(f"   ❌ Error evaluating {params}: {str(e)}")
            return None

    def _acquisition_function(self, gp, param_bounds: List[Tuple], param_names: List[str],
                              param_configs: Dict, n_candidates: int = 1000):
        """
        Acquisition function for Bayesian optimization
        Uses Upper Confidence Bound (UCB) to balance exploration vs exploitation
        """

        # Generate random candidates
        candidates = []
        for _ in range(n_candidates):
            candidate = []
            for (min_val, max_val) in param_bounds:
                candidate.append(np.random.uniform(min_val, max_val))
            candidates.append(candidate)

        candidates = np.array(candidates)

        # Predict mean and uncertainty for all candidates
        mean, std = gp.predict(candidates, return_std=True)

        # UCB acquisition function (mean + exploration_weight * uncertainty)
        exploration_weight = 2.0
        acquisition_values = mean + exploration_weight * std

        # Find the best candidate
        best_idx = np.argmax(acquisition_values)
        best_candidate = candidates[best_idx]

        # Convert back to parameter dictionary
        best_params = {}
        for i, param_name in enumerate(param_names):
            value = best_candidate[i]

            # Round integers
            if param_configs[param_name]["type"] == "integer":
                value = int(round(value))
                # Ensure within bounds
                value = max(param_configs[param_name]["min"],
                            min(param_configs[param_name]["max"], value))

            best_params[param_name] = value

        return best_params, best_candidate.tolist()

    def save_optimization_results(self, filepath: str):
        """Save optimization results to file"""
        results = {
            "strategy_class": self.strategy_class.__name__,
            "optimization_metric": self.optimization_metric,
            "best_parameters": self.best_params,
            "best_score": self.best_score,
            "optimization_history": self.optimization_history,
            "timestamp": datetime.now().isoformat()
        }

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"💾 Optimization results saved to {filepath}")


# Example usage
if __name__ == "__main__":
    # This demonstrates how our CI/CD pipeline will use the optimizer

    print("🚀 Testing AI Parameter Optimizer")
    print("=" * 50)

    # Import our strategy
    import sys

    sys.path.append('../trading_strategies')
    from moving_average_strategy import MovingAverageStrategy

    # Create optimizer
    optimizer = AIParameterOptimizer(MovingAverageStrategy, "sharpe_ratio")

    # Set up tests parameters
    symbol = "AAPL"
    from datetime import datetime, timedelta

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    # Run optimization
    print("\n🔍 Running Grid Search...")
    grid_results = optimizer.grid_search_optimization(
        symbol=symbol,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        max_iterations=20
    )

    # Save results
    optimizer.save_optimization_results("../reports/optimization_results.json")

    print("\n🎯 This is what our CI/CD pipeline will do automatically!")
    print("   ✅ Developer commits strategy")
    print("   ✅ AI finds optimal parameters")
    print("   ✅ Strategy auto-deployed with best settings")
    print("   ✅ Performance monitored continuously")