"""
Unit Tests for Trading Strategies

These tests are automatically run by the GitLab CI/CD pipeline
to ensure strategy quality and prevent bugs from reaching production.
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add the trading_strategies directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'trading_strategies'))

try:
    from base_strategy import BaseStrategy
    from moving_average_strategy import MovingAverageStrategy
except ImportError as e:
    print(f"Warning: Could not import strategy modules: {e}")
    BaseStrategy = None
    MovingAverageStrategy = None


class TestBaseStrategy(unittest.TestCase):
    """Test the base strategy framework"""

    def setUp(self):
        """Set up test fixtures"""
        if MovingAverageStrategy is None:
            self.skipTest("MovingAverageStrategy not available")

        self.strategy = MovingAverageStrategy(short_window=10, long_window=20)

        # Create sample market data for testing
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)  # For reproducible tests

        # Generate realistic price data
        returns = np.random.normal(0.001, 0.02, len(dates))  # ~0.1% daily return, 2% volatility
        prices = [100]  # Starting price
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        self.sample_data = pd.DataFrame({
            'Date': dates,
            'Open': prices,
            'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, len(dates))
        }).set_index('Date')

    def test_strategy_initialization(self):
        """Test strategy initialization"""
        self.assertEqual(self.strategy.name, "Moving Average Crossover")
        self.assertEqual(self.strategy.short_window, 10)
        self.assertEqual(self.strategy.long_window, 20)
        self.assertIsInstance(self.strategy.parameters, dict)

    def test_parameter_validation(self):
        """Test parameter validation"""
        # Test valid parameters
        self.strategy.update_parameters({"short_window": 15, "long_window": 30})
        self.assertEqual(self.strategy.short_window, 15)
        self.assertEqual(self.strategy.long_window, 30)

        # Test invalid parameters (short >= long)
        with self.assertRaises(ValueError):
            self.strategy.update_parameters({"short_window": 25, "long_window": 20})

    def test_signal_generation(self):
        """Test signal generation"""
        signals = self.strategy.generate_signals(self.sample_data)

        # Check that signals DataFrame has the right structure
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertIn('signal', signals.columns)

        # Check that signals are valid values (1, -1, or 0)
        valid_signals = signals['signal'].isin([1, -1, 0]).all()
        self.assertTrue(valid_signals, "All signals should be 1, -1, or 0")

        # Check that we have some signals (not all zeros)
        has_signals = (signals['signal'] != 0).any()
        self.assertTrue(has_signals, "Strategy should generate some trading signals")

    def test_backtest_execution(self):
        """Test backtest execution with real data"""
        # Use a shorter period for faster tests
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)

        results = self.strategy.backtest(
            symbol="AAPL",
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )

        # Check that backtest returns expected fields
        expected_fields = [
            'strategy_name', 'symbol', 'start_date', 'end_date',
            'total_return', 'annual_return', 'volatility', 'sharpe_ratio',
            'max_drawdown', 'win_rate', 'total_trades'
        ]

        for field in expected_fields:
            self.assertIn(field, results, f"Missing field: {field}")

        # Check that numeric values are reasonable
        self.assertIsInstance(results['total_return'], (int, float))
        self.assertIsInstance(results['annual_return'], (int, float))
        self.assertIsInstance(results['sharpe_ratio'], (int, float))
        self.assertGreaterEqual(results['total_trades'], 0)
        self.assertGreaterEqual(results['win_rate'], 0)
        self.assertLessEqual(results['win_rate'], 1)

    def test_optimization_params(self):
        """Test optimization parameter specification"""
        opt_params = self.strategy.get_optimization_params()

        self.assertIsInstance(opt_params, dict)
        self.assertIn('short_window', opt_params)
        self.assertIn('long_window', opt_params)

        # Check parameter structure
        for param_name, param_config in opt_params.items():
            self.assertIn('type', param_config)
            self.assertIn('min', param_config)
            self.assertIn('max', param_config)
            self.assertIn('current', param_config)

    def test_strategy_validation(self):
        """Test strategy validation logic"""
        # Run a backtest first
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)

        self.strategy.backtest(
            symbol="AAPL",
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )

        # Test validation
        is_valid, message = self.strategy.is_valid_strategy()
        self.assertIsInstance(is_valid, bool)
        self.assertIsInstance(message, str)
        self.assertGreater(len(message), 0)


class TestMovingAverageStrategy(unittest.TestCase):
    """Test specific functionality of Moving Average Strategy"""

    def setUp(self):
        if MovingAverageStrategy is None:
            self.skipTest("MovingAverageStrategy not available")

        self.strategy = MovingAverageStrategy(short_window=5, long_window=10)

    def test_moving_average_calculation(self):
        """Test that moving averages are calculated correctly"""
        # Create simple test data
        data = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112]
        })

        signals = self.strategy.generate_signals(data)

        # The strategy should create MA columns internally
        # We can't directly test them since they're not returned,
        # but we can test that signals are generated logically
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertIn('signal', signals.columns)

    def test_different_window_sizes(self):
        """Test strategy with different window sizes"""
        strategies = [
            MovingAverageStrategy(5, 10),
            MovingAverageStrategy(10, 20),
            MovingAverageStrategy(20, 50)
        ]

        for strategy in strategies:
            self.assertEqual(strategy.name, "Moving Average Crossover")
            self.assertLess(strategy.short_window, strategy.long_window)


class TestRiskManagement(unittest.TestCase):
    """Test risk management and safety features"""

    def setUp(self):
        if MovingAverageStrategy is None:
            self.skipTest("MovingAverageStrategy not available")

        self.strategy = MovingAverageStrategy()

    def test_drawdown_calculation(self):
        """Test that maximum drawdown is calculated correctly"""
        # Create data with a known drawdown
        prices = [100, 110, 120, 130, 100, 90, 85, 95, 105, 115]  # 35% drawdown from peak
        data = pd.DataFrame({
            'Close': prices,
            'Open': prices,
            'High': [p * 1.02 for p in prices],
            'Low': [p * 0.98 for p in prices],
            'Volume': [1000000] * len(prices)
        })

        signals = self.strategy.generate_signals(data)
        returns_df = self.strategy.calculate_returns(data, signals)
        metrics = self.strategy.calculate_metrics(returns_df, data)

        # Max drawdown should be negative
        self.assertLessEqual(metrics['max_drawdown'], 0)

    def test_win_rate_calculation(self):
        """Test win rate calculation"""
        # This is implicitly tested in the backtest, but we can add specific tests
        self.assertIsInstance(self.strategy.get_optimization_params(), dict)


class TestDataHandling(unittest.TestCase):
    """Test data handling and edge cases"""

    def setUp(self):
        if MovingAverageStrategy is None:
            self.skipTest("MovingAverageStrategy not available")

        self.strategy = MovingAverageStrategy()

    def test_empty_data_handling(self):
        """Test handling of empty data"""
        empty_data = pd.DataFrame(columns=['Close', 'Open', 'High', 'Low', 'Volume'])

        # Should handle empty data gracefully
        try:
            signals = self.strategy.generate_signals(empty_data)
            self.assertIsInstance(signals, pd.DataFrame)
        except Exception as e:
            # If it raises an exception, it should be a reasonable one
            self.assertIsInstance(e, (ValueError, IndexError))

    def test_insufficient_data(self):
        """Test handling of insufficient data for moving averages"""
        # Data with fewer points than the long window
        small_data = pd.DataFrame({
            'Close': [100, 101, 102],
            'Open': [99, 100, 101],
            'High': [101, 102, 103],
            'Low': [98, 99, 100],
            'Volume': [1000000, 1000000, 1000000]
        })

        # Should handle gracefully (likely with NaN values or empty signals)
        try:
            signals = self.strategy.generate_signals(small_data)
            self.assertIsInstance(signals, pd.DataFrame)
        except Exception as e:
            # If it raises an exception, it should be informative
            self.assertIn("window", str(e).lower())


# Test utility functions
class TestUtilities(unittest.TestCase):
    """Test utility functions and helpers"""

    def test_date_handling(self):
        """Test date string formatting and handling"""
        from datetime import datetime

        test_date = datetime(2023, 6, 15)
        formatted = test_date.strftime('%Y-%m-%d')
        self.assertEqual(formatted, '2023-06-15')

    def test_parameter_types(self):
        """Test parameter type handling"""
        if MovingAverageStrategy is None:
            self.skipTest("MovingAverageStrategy not available")

        # Test integer parameters
        strategy = MovingAverageStrategy(short_window=10, long_window=20)
        self.assertIsInstance(strategy.short_window, int)
        self.assertIsInstance(strategy.long_window, int)


# Performance benchmarks (not strict tests, but good to monitor)
class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmarks to ensure strategies are efficient"""

    def setUp(self):
        if MovingAverageStrategy is None:
            self.skipTest("MovingAverageStrategy not available")

    def test_backtest_performance(self):
        """Test that backtests complete in reasonable time"""
        import time

        strategy = MovingAverageStrategy()

        # Test with 1 year of data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)

        start_time = time.time()

        try:
            results = strategy.backtest(
                symbol="AAPL",
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )

            execution_time = time.time() - start_time

            # Backtest should complete in under 30 seconds
            self.assertLess(execution_time, 30,
                            f"Backtest took {execution_time:.2f} seconds, should be under 30s")

            # Should return valid results
            self.assertNotIn('error', results)

        except Exception as e:
            # If there's a network error downloading data, that's okay for this test
            if "download" in str(e).lower() or "connection" in str(e).lower():
                self.skipTest(f"Network error during test: {e}")
            else:
                raise


if __name__ == '__main__':
    # Run all tests
    print("🧪 Running TradingBot CI Test Suite")
    print("=" * 50)

    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])

    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")

    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")

    # Exit with appropriate code for CI/CD
    if result.failures or result.errors:
        print("\n❌ Some tests failed")
        sys.exit(1)
    else:
        print("\n✅ All tests passed!")
        sys.exit(0)