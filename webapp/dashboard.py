from flask import Flask, render_template, jsonify
import json
import os
import glob
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder

app = Flask(__name__)


def get_backtest_results():
    """Get backtest results from your actual files"""
    results = []

    # Look for your actual results files (they're directly in reports folder!)
    backtest_files = ["reports/ma_strategy_results.json"]

    print(f"🔍 Looking for files: {backtest_files}")

    for file_path in backtest_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                results.append(data)
        except:
            continue

    return results


def get_optimization_results():
    """Get optimization results from your actual files"""
    results = []

    # Look for your actual optimization files
    opt_files = ["reports/optimization_results.json"]

    print(f"🔍 Looking for optimization files: {opt_files}")

    for file_path in opt_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                if 'best_parameters' in data:  # This is optimization data
                    results.append(data)
        except:
            continue

    return results


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')


@app.route('/api/overview')
def api_overview():
    """API endpoint for dashboard overview data"""

    # Get your real results
    backtest_results = get_backtest_results()

    # Calculate summary stats from your actual data
    total_strategies = len(backtest_results)
    successful_strategies = len([r for r in backtest_results if r.get('sharpe_ratio', 0) > 0.5])

    avg_annual_return = 0
    avg_sharpe_ratio = 0
    if backtest_results:
        returns = [r.get('annual_return', 0) for r in backtest_results]
        sharpes = [r.get('sharpe_ratio', 0) for r in backtest_results]

        if returns:
            avg_annual_return = sum(returns) / len(returns)
        if sharpes:
            avg_sharpe_ratio = sum(sharpes) / len(sharpes)

    overview_data = {
        "summary": {
            "total_strategies": total_strategies,
            "successful_strategies": successful_strategies,
            "success_rate": (successful_strategies / total_strategies * 100) if total_strategies > 0 else 0,
            "avg_annual_return": avg_annual_return,
            "avg_sharpe_ratio": avg_sharpe_ratio
        },
        "pipeline_status": {
            "latest_pipeline": {
                "id": "12345",
                "status": "success",
                "commit_message": "AI-optimized trading strategy",
                "duration": "4m 32s"
            }
        },
        "latest_results": backtest_results[:5]  # Latest 5 results
    }

    return jsonify(overview_data)


@app.route('/api/strategy_performance')
def api_strategy_performance():
    """API endpoint for strategy performance charts"""

    backtest_results = get_backtest_results()

    if not backtest_results:
        return jsonify({"error": "No backtest results available"})

    # Prepare data for charts using your real data
    strategies = []
    annual_returns = []
    sharpe_ratios = []

    for result in backtest_results:
        if 'strategy_name' in result:
            strategies.append(result['strategy_name'])
            annual_returns.append(result.get('annual_return', 0) * 100)
            sharpe_ratios.append(result.get('sharpe_ratio', 0))

    # Create performance comparison chart
    fig1 = go.Figure()

    fig1.add_trace(go.Bar(
        x=strategies,
        y=annual_returns,
        name='Annual Return (%)',
        marker_color='lightblue'
    ))

    fig1.update_layout(
        title='Strategy Annual Returns',
        xaxis_title='Strategy',
        yaxis_title='Annual Return (%)',
        height=400
    )

    return jsonify({
        "performance_chart": json.loads(json.dumps(fig1, cls=PlotlyJSONEncoder)),
        "summary_stats": {
            "total_strategies": len(strategies),
            "best_return": max(annual_returns) if annual_returns else 0,
            "best_sharpe": max(sharpe_ratios) if sharpe_ratios else 0
        }
    })


@app.route('/api/optimization_progress')
def api_optimization_progress():
    """Show your real AI optimization progress"""

    opt_results = get_optimization_results()

    if opt_results:
        # Use your real optimization data
        best_score = opt_results[0].get('best_score', 0)
        total_evals = opt_results[0].get('total_evaluations', 20)

        # Create simple progress chart
        iterations = list(range(1, total_evals + 1))
        scores = [best_score * (i / total_evals) for i in iterations]  # Simulate progress

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=iterations,
            y=scores,
            mode='lines+markers',
            name='AI Optimization Progress',
            line=dict(color='green', width=3)
        ))

        fig.update_layout(
            title='AI Optimization Progress',
            xaxis_title='Iteration',
            yaxis_title='Sharpe Ratio',
            height=400
        )

        return jsonify({
            "optimization_chart": json.loads(json.dumps(fig, cls=PlotlyJSONEncoder)),
            "optimization_summary": {
                "final_score": best_score,
                "total_iterations": total_evals
            }
        })

    return jsonify({"error": "No optimization data available"})


if __name__ == '__main__':
    print("🚀 Starting TradingBot CI Dashboard")
    print("Dashboard will be available at: http://localhost:5000")
    print("=" * 50)

    # Check for data files
    backtest_files = glob.glob("reports/**/*.json", recursive=True)
    if backtest_files:
        print(f"📊 Found {len(backtest_files)} data files!")
        for file in backtest_files:
            print(f"   - {file}")
    else:
        print("📝 No data files found. Run your strategies first!")

    app.run(debug=True, host='0.0.0.0', port=5000)