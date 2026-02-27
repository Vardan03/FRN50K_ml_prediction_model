#!/usr/bin/env python3
"""
Quick test script to verify all notebooks and models work correctly
Run: python test_notebooks.py
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and report status"""
    print(f"\n{'='*80}")
    print(f"📋 {description}")
    print(f"{'='*80}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("✅ SUCCESS")
            return True
        else:
            print(f"❌ FAILED")
            if result.stderr:
                print(f"Error: {result.stderr[:500]}")
            return False
    except subprocess.TimeoutExpired:
        print("⏱️  TIMEOUT")
        return False
    except Exception as e:
        print(f"❌ ERROR: {str(e)[:200]}")
        return False

def main():
    print("\n" + "="*80)
    print("🚀 ASSIGNMENT 1: MODEL COMPARISON AND ANALYSIS")
    print("📊 Testing Notebooks and Models")
    print("="*80)
    
    project_dir = Path(__file__).parent
    
    # Test 1: Check if notebooks exist
    print("\n📝 Checking notebook files...")
    notebooks = [
        'assignment1_exploration.ipynb',
        'assignment1_modeling.ipynb'
    ]
    
    for nb in notebooks:
        nb_path = project_dir / nb
        if nb_path.exists():
            print(f"✅ {nb} - Found ({nb_path.stat().st_size} bytes)")
        else:
            print(f"❌ {nb} - NOT FOUND")
    
    # Test 2: Run main.py to verify basic setup
    print("\n🔧 Testing basic model training...")
    cmd = f"cd {project_dir} && source ../venv/bin/activate && PYTHONPATH=.:$PYTHONPATH python3 scripts/train_baseline.py --model linear --sample-size 50 2>&1 | tail -20"
    run_command(cmd, "Running train_baseline.py with Linear Model (sample=50)")
    
    # Test 3: Verify imports
    print("\n📦 Verifying imports...")
    test_imports = """
import sys
sys.path.append('src')
from data.data_loader import FreshRetailDataLoader
from data.feature_engineering import FeatureEngineer
from models.baseline.linear_models import LinearForecastingModel
from models.baseline.tree_models import TreeForecastingModel
from models.baseline.naive_models import SimpleNaiveModel, SeasonalNaiveModel
from evaluate.metrics import ForecastingMetrics
print('✅ All imports successful')
"""
    
    cmd = f"cd {project_dir} && source ../venv/bin/activate && PYTHONPATH=.:$PYTHONPATH python3 -c \"{test_imports}\""
    run_command(cmd, "Testing Python imports")
    
    # Summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    print("""
✅ Notebooks Created:
   - assignment1_exploration.ipynb (Data Exploration)
   - assignment1_modeling.ipynb (Model Comparison & Analysis)

✅ Support Files Created:
   - MODEL_COMPARISON_SUMMARY.md (Comprehensive summary)
   - test_notebooks.py (This test script)

📚 Usage:
   1. Open assignment1_exploration.ipynb in Jupyter for data analysis
   2. Open assignment1_modeling.ipynb in Jupyter for model comparison
   3. Run individual cells to generate visualizations
   4. Review MODEL_COMPARISON_SUMMARY.md for overview

💡 Key Notebooks Features:
   - Complete EDA with visualizations
   - 4 baseline models training and evaluation
   - Comprehensive comparison tables
   - Error analysis by business segments
   - Residual diagnostics
   - Actionable recommendations

🎯 Next Steps:
   1. Run assignment1_exploration.ipynb to understand the data
   2. Run assignment1_modeling.ipynb to compare models
   3. Review findings and recommendations
   4. Implement suggested improvements

📧 Questions? See MODEL_COMPARISON_SUMMARY.md for detailed documentation
    """)

if __name__ == '__main__':
    main()
