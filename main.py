import os
import sys

# Add the current directory to Python path so we can import our modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Add webapp directory to path since dashboard.py is there
webapp_dir = os.path.join(current_dir, 'webapp')
if os.path.exists(webapp_dir):
    sys.path.insert(0, webapp_dir)

try:
    # Try to import from webapp folder first
    from webapp.dashboard import app

    print("✅ Imported dashboard from webapp folder")
except ImportError:
    try:
        # If that fails, try importing from root
        from dashboard import app

        print("✅ Imported dashboard from root folder")
    except ImportError:
        print("❌ Could not import dashboard module")
        print("Available files:", os.listdir('.'))
        if os.path.exists('webapp'):
            print("Files in webapp:", os.listdir('webapp'))
        sys.exit(1)

if __name__ == "__main__":
    print("🚀 Starting TradingBot CI Dashboard")
    print("=" * 50)

    # Get port from environment (for deployment) or use 5001 for local
    port = int(os.environ.get('PORT', 5001))

    print(f"🌐 Dashboard will run on port {port}")
    print("🔗 Local URL: http://localhost:5001")

    # Run the Flask app
    app.run(
        host='0.0.0.0',  # Allow external connections
        port=port,
        debug=False  # Turn off debug mode for production
    )
