from moving_average_strategy import MovingAverageStrategy
from datetime import datetime, timedelta

# Test the AI-optimized parameters (replace with actual values from optimization_results.json)
print("🤖 Testing AI-Optimized Parameters")
print("=" * 50)

# Original losing strategy
original_strategy = MovingAverageStrategy(short_window=20, long_window=50)
print("📉 Original Strategy (20/50):")

end_date = datetime.now()
start_date = end_date - timedelta(days=365)

original_results = original_strategy.backtest(
    symbol="AAPL",
    start_date=start_date.strftime('%Y-%m-%d'),
    end_date=end_date.strftime('%Y-%m-%d')
)

print(f"   Annual Return: {original_results['annual_return']:.2%}")
print(f"   Sharpe Ratio: {original_results['sharpe_ratio']:.2f}")

print("\n" + "="*50)

# AI-optimized strategy (replace these with values from your optimization results)
optimized_strategy = MovingAverageStrategy(short_window=12, long_window=35)  # Example values
print("🚀 AI-Optimized Strategy (12/35):")

optimized_results = optimized_strategy.backtest(
    symbol="AAPL",
    start_date=start_date.strftime('%Y-%m-%d'),
    end_date=end_date.strftime('%Y-%m-%d')
)

print(f"   Annual Return: {optimized_results['annual_return']:.2%}")
print(f"   Sharpe Ratio: {optimized_results['sharpe_ratio']:.2f}")

# Calculate improvement
improvement = optimized_results['annual_return'] - original_results['annual_return']
print(f"\n🎯 AI Improvement: {improvement:.2%} better annual return!")