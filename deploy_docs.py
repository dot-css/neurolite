#!/usr/bin/env python3
"""
NeuroLite Documentation Deployment Script

This script helps deploy NeuroLite documentation to Read the Docs.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(cmd, cwd=None):
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            cwd=cwd, 
            capture_output=True, 
            text=True, 
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}")
        print(f"Error: {e.stderr}")
        return None

def check_dependencies():
    """Check if required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'sphinx',
        'sphinx-rtd-theme',
        'myst-parser',
        'sphinx-autodoc-typehints',
        'sphinx-copybutton'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ❌ {package}")
    
    if missing_packages:
        print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
        cmd = f"pip install {' '.join(missing_packages)}"
        if run_command(cmd):
            print("  ✅ Dependencies installed successfully")
        else:
            print("  ❌ Failed to install dependencies")
            return False
    
    return True

def build_docs():
    """Build the documentation using Sphinx."""
    print("\n🏗️  Building documentation...")
    
    docs_dir = Path("docs")
    build_dir = docs_dir / "_build"
    
    # Clean previous build
    if build_dir.exists():
        shutil.rmtree(build_dir)
        print("  🧹 Cleaned previous build")
    
    # Build HTML documentation
    cmd = "sphinx-build -b html . _build/html"
    result = run_command(cmd, cwd=docs_dir)
    
    if result is not None:
        print("  ✅ Documentation built successfully")
        print(f"  📁 Output directory: {build_dir / 'html'}")
        return True
    else:
        print("  ❌ Failed to build documentation")
        return False

def validate_build():
    """Validate the built documentation."""
    print("\n🔍 Validating build...")
    
    build_dir = Path("docs/_build/html")
    
    # Check if index.html exists
    index_file = build_dir / "index.html"
    if index_file.exists():
        print("  ✅ index.html found")
    else:
        print("  ❌ index.html not found")
        return False
    
    # Check for common files
    expected_files = [
        "getting_started/installation.html",
        "getting_started/quickstart.html",
        "api/core.html",
        "tutorials/image_classification.html",
        "faq.html",
        "troubleshooting.html"
    ]
    
    for file_path in expected_files:
        full_path = build_dir / file_path
        if full_path.exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ⚠️  {file_path} not found")
    
    return True

def create_github_workflow():
    """Create GitHub Actions workflow for automatic documentation deployment."""
    print("\n🔧 Creating GitHub Actions workflow...")
    
    workflow_dir = Path(".github/workflows")
    workflow_dir.mkdir(parents=True, exist_ok=True)
    
    workflow_content = """name: Documentation

on:
  push:
    branches: [ main, master ]
  pull_request:
    branches: [ main, master ]

jobs:
  docs:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[docs]"
        pip install -r docs/requirements.txt
    
    - name: Build documentation
      run: |
        cd docs
        sphinx-build -b html . _build/html
    
    - name: Deploy to GitHub Pages
      if: github.ref == 'refs/heads/main'
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./docs/_build/html
"""
    
    workflow_file = workflow_dir / "docs.yml"
    with open(workflow_file, 'w') as f:
        f.write(workflow_content)
    
    print(f"  ✅ Created workflow: {workflow_file}")

def setup_readthedocs():
    """Setup Read the Docs configuration."""
    print("\n📚 Setting up Read the Docs configuration...")
    
    # Check if .readthedocs.yaml exists
    rtd_config = Path(".readthedocs.yaml")
    if rtd_config.exists():
        print("  ✅ .readthedocs.yaml already exists")
    else:
        print("  ❌ .readthedocs.yaml not found")
        return False
    
    # Check docs/requirements.txt
    docs_requirements = Path("docs/requirements.txt")
    if docs_requirements.exists():
        print("  ✅ docs/requirements.txt exists")
    else:
        print("  ❌ docs/requirements.txt not found")
        return False
    
    print("\n📋 Read the Docs setup checklist:")
    print("  1. Go to https://readthedocs.org/")
    print("  2. Sign in with your GitHub account")
    print("  3. Import your repository")
    print("  4. The build should start automatically")
    print("  5. Your docs will be available at: https://neurolite.readthedocs.io/")
    
    return True

def serve_docs():
    """Serve documentation locally for testing."""
    print("\n🌐 Starting local documentation server...")
    
    build_dir = Path("docs/_build/html")
    if not build_dir.exists():
        print("  ❌ Documentation not built. Run build first.")
        return False
    
    try:
        import http.server
        import socketserver
        import webbrowser
        
        PORT = 8000
        os.chdir(build_dir)
        
        Handler = http.server.SimpleHTTPRequestHandler
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print(f"  🚀 Serving at http://localhost:{PORT}")
            print("  📖 Opening documentation in browser...")
            webbrowser.open(f"http://localhost:{PORT}")
            print("  ⏹️  Press Ctrl+C to stop the server")
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\n  ✅ Server stopped")
    except Exception as e:
        print(f"  ❌ Error starting server: {e}")

def main():
    """Main deployment function."""
    print("🧠⚡ NeuroLite Documentation Deployment")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
    else:
        print("\nAvailable commands:")
        print("  check     - Check dependencies")
        print("  build     - Build documentation")
        print("  serve     - Serve documentation locally")
        print("  deploy    - Full deployment setup")
        print("  workflow  - Create GitHub Actions workflow")
        print("  rtd       - Setup Read the Docs")
        print("\nUsage: python deploy_docs.py <command>")
        return
    
    if command == "check":
        check_dependencies()
    
    elif command == "build":
        if check_dependencies():
            build_docs()
            validate_build()
    
    elif command == "serve":
        serve_docs()
    
    elif command == "deploy":
        print("🚀 Full deployment setup...")
        if check_dependencies():
            if build_docs():
                validate_build()
                create_github_workflow()
                setup_readthedocs()
                print("\n✅ Deployment setup complete!")
                print("\nNext steps:")
                print("1. Commit and push your changes to GitHub")
                print("2. Set up your project on Read the Docs")
                print("3. Your documentation will be live!")
    
    elif command == "workflow":
        create_github_workflow()
    
    elif command == "rtd":
        setup_readthedocs()
    
    else:
        print(f"❌ Unknown command: {command}")
        print("Run 'python deploy_docs.py' to see available commands")

if __name__ == "__main__":
    main()