import os
import sys

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Add webapp directory to path since dashboard.py is there
webapp_dir = os.path.join(current_dir, 'webapp')
sys.path.insert(0, webapp_dir)

try:
    # Import from webapp folder (where dashboard.py actually is)
    from webapp.dashboard import app
    print("✅ Successfully imported dashboard from webapp folder")
except ImportError:
    try:
        # Fallback: try direct import if in same directory
        from dashboard import app
        print("✅ Successfully imported dashboard from current directory")
    except ImportError as e:
        print(f"❌ Could not import dashboard module: {e}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Available files: {os.listdir('.')}")
        if os.path.exists('webapp'):
            print(f"Files in webapp: {os.listdir('webapp')}")
        sys.exit(1)

if __name__ == "__main__":
    # Railway provides PORT environment variable
    port = int(os.environ.get('PORT', 8000))
    
    print(f"🚀 Starting TradingBot CI on port {port}")
    print(f"📁 Working directory: {os.getcwd()}")
    
    # Run the app
    app.run(
        host='0.0.0.0',  # Railway requires this
        port=port,       # Railway provides this
        debug=False      # No debug in production
    )
