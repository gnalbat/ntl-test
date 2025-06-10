#!/usr/bin/env python3
"""
Simple test script to validate VIIRS data access and basic processing
"""

import ee
import yaml

def test_basic_viirs_access():
    """Test basic VIIRS data access"""
    
    # Load config
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # Initialize Earth Engine
    ee.Initialize(project=config['ee']['project'])
    
    # Define ROI
    roi = ee.Geometry.Rectangle(config['ee']['roi'])
    
    print("Testing VIIRS data access...")
    print(f"ROI: {config['ee']['roi']}")
    print(f"Project: {config['ee']['project']}")
    
    # Test collection access
    collection = ee.ImageCollection(config['ee']['ntl_collection_id']) \
        .filterDate('2024-01-01', '2024-01-31') \
        .filterBounds(roi)
    
    size = collection.size().getInfo()
    print(f"Found {size} images in January 2024")
    
    if size > 0:
        # Get first image
        first_image = ee.Image(collection.first())
        bands = first_image.bandNames().getInfo()
        print(f"Available bands: {bands}")
        
        # Test simple mean calculation (no masking)
        mean_image = collection.select(config['ee']['ntl_band']).mean()
        
        # Test basic statistics
        stats = mean_image.reduceRegion(
            reducer=ee.Reducer.mean().combine(ee.Reducer.minMax(), sharedInputs=True),
            geometry=roi,
            scale=1000,  # Use larger scale for faster processing
            maxPixels=1e8,
            bestEffort=True
        ).getInfo()
        
        print(f"Basic statistics (no masking):")
        for key, value in stats.items():
            print(f"  {key}: {value}")
            
        print("✓ Basic VIIRS access test passed!")
        return True
        
    else:
        print("✗ No images found")
        return False

if __name__ == "__main__":
    try:
        test_basic_viirs_access()
    except Exception as e:
        print(f"✗ Test failed: {e}")
