"""Demo script for feature engineering following Transformer notebook patterns.

This script demonstrates the feature engineering pipeline with sample data,
similar to the "Test Execution" section in the Transformer notebook.
"""

import pandas as pd
import numpy as np
from src.feature_engineering import engineer_features, StoreNameExtractor

def demo_store_name_extraction():
    """Demo store name extraction functionality."""
    print("=" * 60)
    print("🏪 STORE NAME EXTRACTION DEMO")
    print("=" * 60)
    
    # Initialize extractor
    extractor = StoreNameExtractor()
    
    # Test cases
    test_cases = [
        "SQ *WHOLE FOODS MARKET",
        "AMAZON MKTP US*1234567",
        "UBER TRIP 123456",
        "LYFT RIDE FARE",
        "REGULAR STORE NAME",
        "WALMART SUPERCENTER",
        "CVS PHARMACY #1234",
    ]
    
    for description in test_cases:
        store_name = extractor(description)
        print(f"  '{description}' -> '{store_name}'")
    
    print()


def demo_feature_engineering():
    """Demo complete feature engineering pipeline."""
    print("=" * 60)
    print("🔧 FEATURE ENGINEERING PIPELINE DEMO")
    print("=" * 60)
    
    # Create sample transaction data
    sample_data = pd.DataFrame({
        'transaction_id': [1, 2, 3, 4, 5],
        'date': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04']),
        'description': [
            'SQ *WHOLE FOODS MARKET',
            'AMAZON MKTP US*1234567',
            'UBER TRIP 123456',
            'LYFT RIDE FARE',
            'REGULAR STORE NAME'
        ],
        'amount': [42.75, 25.50, 15.20, 18.30, 100.00],
        'day_of_week': ['Sunday', 'Sunday', 'Monday', 'Tuesday', 'Wednesday']
    })
    
    print("Input data:")
    print(sample_data)
    print(f"\nInput shape: {sample_data.shape}")
    
    # Apply feature engineering
    try:
        engineered = engineer_features(sample_data)
        print(f"\nEngineered features shape: {engineered.shape}")
        print(f"Number of features created: {engineered.shape[1] - sample_data.shape[1]}")
        
        # Show some key features
        key_features = ['store_name', 'is_round_amount', 'decimal_digits', 
                      'has_grocery_keyword', 'has_transport_keyword',
                      'day_of_year', 'is_weekend']
        
        print(f"\nKey engineered features:")
        print(engineered[key_features])
        
        # Show feature statistics
        print(f"\nFeature statistics:")
        for feature in key_features:
            if feature in engineered.columns:
                unique_vals = engineered[feature].nunique()
                print(f"  {feature}: {unique_vals} unique values")
        
        print("\n✅ Feature engineering completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during feature engineering: {e}")
        raise
    
    print()


def demo_error_handling():
    """Demo error handling and validation."""
    print("=" * 60)
    print("🚨 ERROR HANDLING DEMO")
    print("=" * 60)
    
    # Test missing required column
    try:
        bad_data = pd.DataFrame({
            'transaction_id': [1, 2, 3],
            'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
            # Missing 'description', 'amount', 'day_of_week'
        })
        
        engineered = engineer_features(bad_data)
        print("❌ Should have failed but didn't!")
        
    except ValueError as e:
        print(f"✅ Correctly caught missing columns error:")
        print(f"   {e}")
    
    # Test invalid date format
    try:
        bad_data = pd.DataFrame({
            'transaction_id': [1, 2, 3],
            'date': ['invalid', 'dates', 'here'],  # Invalid dates
            'description': ['A', 'B', 'C'],
            'amount': [10, 20, 30],
            'day_of_week': ['Mon', 'Tue', 'Wed']
        })
        
        engineered = engineer_features(bad_data)
        print("❌ Should have failed but didn't!")
        
    except ValueError as e:
        print(f"\n✅ Correctly caught invalid date error:")
        print(f"   {e}")
    
    print()


def main():
    """Run all demos."""
    print("🧪 AURORAGATE FEATURE ENGINEERING DEMOS")
    print("Following Transformer notebook patterns")
    print()
    
    demo_store_name_extraction()
    demo_feature_engineering()
    demo_error_handling()
    
    print("=" * 60)
    print("🎉 ALL DEMOS COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()