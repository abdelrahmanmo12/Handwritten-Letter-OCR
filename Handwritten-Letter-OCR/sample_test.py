import sys
import os

# Add src to path
sys.path.append('src')

def test_imports():
    print("🔧 Testing imports after fix...")
    
    modules = [
        'data_loader',
        'preprocessing', 
        'feature_extraction',
        'models',
        'evaluation',
        'gui',
        'sample_data_loader'
    ]
    
    all_ok = True
    for module in modules:
        try:
            __import__(module)
            print(f"✅ {module} - OK")
        except ImportError as e:
            print(f"❌ {module} - FAILED: {e}")
            all_ok = False
    
    return all_ok

if __name__ == "__main__":
    if test_imports():
        print("\n🎉 All imports fixed! Now run: python run_training.py")
    else:
        print("\n❌ Some imports still failing.")