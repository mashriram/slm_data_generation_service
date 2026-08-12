import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

try:
    from app.main import app
    print("Successfully imported app")
    
    # Check for critical routes
    routes = [route.path for route in app.routes]
    required_routes = ["/api/v1/tokens", "/api/v1/generate", "/api/v1/dataset/modify"]
    
    missing = []
    for req in required_routes:
        if not any(req in r for r in routes):
            missing.append(req)
            
    if missing:
        print(f"Missing routes: {missing}")
        sys.exit(1)
        
    print("All critical routes found.")
    print("Startup check passed")
    
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Startup Error: {e}")
    sys.exit(1)
