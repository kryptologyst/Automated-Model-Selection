#!/usr/bin/env python3
"""Quick start script for automated model selection."""

import subprocess
import sys
from pathlib import Path


def main():
    """Main function to run quick start demo."""
    print("🤖 Automated Model Selection - Quick Start")
    print("=" * 50)
    
    # Check if package is installed
    try:
        import automated_model_selection
        print("✅ Package is installed")
    except ImportError:
        print("❌ Package not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
        print("✅ Package installed successfully")
    
    # Run the demo
    print("\n🚀 Running automated model selection demo...")
    demo_script = Path(__file__).parent / "scripts" / "run_demo.py"
    
    if demo_script.exists():
        subprocess.run([sys.executable, str(demo_script)], check=True)
    else:
        print("❌ Demo script not found")
        return 1
    
    print("\n📚 Next Steps:")
    print("1. Run interactive demo: streamlit run demo/streamlit_demo.py")
    print("2. Explore the code in src/automated_model_selection/")
    print("3. Check out the README.md for detailed documentation")
    print("4. Run tests: pytest tests/")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
