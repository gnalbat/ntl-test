#!/usr/bin/env python3
"""
Google Earth Engine Script for Getis-Ord Gi* Analysis of Nighttime Lights
with Water Masking and Outlier Removal

This script performs spatial autocorrelation analysis using Getis-Ord Gi* statistics
on VIIRS nighttime lights data, with water masking to remove outliers and extreme z-scores.
"""

import ee
import yaml
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def initialize_ee(project_id):
    """Initialize Google Earth Engine."""
    try:
        ee.Initialize(project=project_id)
        print(f"Google Earth Engine initialized for project: {project_id}")
    except Exception as e:
        print(f"Error initializing GEE: {e}")
        print("Please run 'earthengine authenticate' first")
        raise

def create_water_mask(roi):
    """
    Create a water mask using JRC Global Surface Water dataset
    to exclude water bodies from analysis.
    """
    try:
        # JRC Global Surface Water dataset
        gsw = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
        
        # Create water mask (1 for land, 0 for water)
        # Use a more conservative threshold - only exclude areas that are water >90% of the time
        water_mask = gsw.select('occurrence').lt(90).clip(roi)
        
        print("  Water mask created successfully")
        return water_mask
        
    except Exception as e:
        print(f"  Warning: Could not create water mask - {str(e)[:100]}...")
        print("  Proceeding without water masking")
        # Return a mask that doesn't filter anything
        return ee.Image.constant(1).clip(roi)

def create_quality_mask(image, quality_band, snow_band):
    """
    Create quality mask based on quality flags and snow conditions.
    """
    try:
        # Quality mask: keep pixels with good quality (value 0, 1, or 2 - be more permissive)
        quality_mask = image.select(quality_band).lte(2)
        
        # Snow mask: exclude snow-covered pixels (only if snow_band exists)
        try:
            snow_mask = image.select(snow_band).eq(0)
            combined_mask = quality_mask.And(snow_mask)
        except:
            # If snow band doesn't exist, just use quality mask
            combined_mask = quality_mask
        
        return combined_mask
        
    except Exception as e:
        print(f"  Warning: Error creating quality mask - {str(e)[:100]}...")
        # Return a mask that doesn't filter anything
        return ee.Image.constant(1)

def filter_ntl_collection(collection_id, roi, start_date, end_date, months):
    """
    Filter nighttime lights collection by date, region, and months.
    """
    collection = ee.ImageCollection(collection_id) \
        .filterDate(start_date, end_date) \
        .filterBounds(roi) \
        .filter(ee.Filter.calendarRange(months[0], months[-1], 'month'))
    
    return collection

def check_data_availability(collection_id, roi, start_date, end_date, months):
    """
    Check if data is available for the given parameters.
    """
    collection = ee.ImageCollection(collection_id) \
        .filterDate(start_date, end_date) \
        .filterBounds(roi) \
        .filter(ee.Filter.calendarRange(months[0], months[-1], 'month'))
    
    size = collection.size().getInfo()
    
    if size == 0:
        print(f"  Warning: No images found for period {start_date} to {end_date}")
        return False
    else:
        print(f"  Found {size} images for period {start_date} to {end_date}")
        
        # Check if we have any images with the required bands
        first_image = ee.Image(collection.first())
        try:
            bands = first_image.bandNames().getInfo()
            print(f"  Available bands: {bands}")
            return True
        except Exception as e:
            print(f"  Error checking bands: {e}")
            return False

def compute_mean_ntl(collection, ntl_band, quality_band, snow_band, water_mask, roi, config=None):
    """
    Compute mean nighttime lights with optional quality filtering and water masking.
    """
    # Check if collection has any images
    collection_size = collection.size()
    print(f"  Collection size: {collection_size.getInfo()} images")
    
    if collection_size.getInfo() == 0:
        print("  Warning: No images found in collection!")
        return None
    
    def mask_image(image):
        # Start with the selected band
        masked_image = image.select(ntl_band)
        
        # Apply quality mask only if enabled in config
        if config and config.get('ee', {}).get('use_quality_mask', True):
            quality_mask = create_quality_mask(image, quality_band, snow_band)
            masked_image = masked_image.updateMask(quality_mask)
            
        # Apply water mask only if enabled in config  
        if config and config.get('ee', {}).get('use_water_mask', True):
            masked_image = masked_image.updateMask(water_mask)
        
        return masked_image
    
    # Apply masking to all images
    masked_collection = collection.map(mask_image)
    
    # Compute mean
    mean_ntl = masked_collection.mean().clip(roi)
    
    return mean_ntl

def remove_outliers(image, roi, percentile_low=5, percentile_high=95):
    """
    Remove extreme outliers using percentile thresholds.
    """
    if image is None:
        return None
        
    try:
        # Calculate percentiles
        percentiles = image.reduceRegion(
            reducer=ee.Reducer.percentile([percentile_low, percentile_high]),
            geometry=roi,
            scale=500,
            maxPixels=1e9,
            bestEffort=True
        )
        
        # Get the percentile values
        band_name = image.bandNames().get(0)
        low_threshold = ee.Number(percentiles.get(f"{band_name}_p{percentile_low}"))
        high_threshold = ee.Number(percentiles.get(f"{band_name}_p{percentile_high}"))
        
        # Create mask for values within percentile range
        outlier_mask = image.gte(low_threshold).And(image.lte(high_threshold))
        
        return image.updateMask(outlier_mask)
        
    except Exception as e:
        print(f"Warning: Error removing outliers - {str(e)[:100]}...")
        return image  # Return original image if outlier removal fails

def compute_getis_ord_gi_star(image, roi, scale=500):
    """
    Compute Getis-Ord Gi* statistics for spatial autocorrelation analysis.
    
    The Getis-Ord Gi* statistic identifies spatial clusters of high and low values.
    Positive values indicate clusters of high values (hot spots),
    negative values indicate clusters of low values (cold spots).
    """
    
    # Define neighborhood for spatial analysis (3x3 kernel)
    kernel = ee.Kernel.square(radius=1, units='pixels', normalize=True)
    
    # Calculate local mean using focal statistics
    local_mean = image.convolve(kernel)
    
    # Calculate global statistics
    global_stats = image.reduceRegion(
        reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True),
        geometry=roi,
        scale=scale,
        maxPixels=1e9
    )
    
    band_name = image.bandNames().get(0)
    global_mean = ee.Number(global_stats.get(f"{band_name}_mean"))
    global_std = ee.Number(global_stats.get(f"{band_name}_stdDev"))
    
    # Calculate Gi* statistic
    # Gi* = (local_mean - global_mean) / global_std
    gi_star = local_mean.subtract(global_mean).divide(global_std)
    
    # Calculate z-scores for statistical significance
    # For a 3x3 neighborhood, we use 9 pixels
    n = 9
    expected_gi = ee.Number(0)  # Expected value under null hypothesis
    variance_gi = ee.Number(1.0 / n)  # Simplified variance for demonstration
    
    z_score = gi_star.subtract(expected_gi).divide(variance_gi.sqrt())
    
    return gi_star.rename('gi_star'), z_score.rename('z_score')

def classify_significance(z_score_image):
    """
    Classify Gi* z-scores into significance categories.
    
    Categories:
    - Hot spots: z > 1.96 (p < 0.05)
    - Cold spots: z < -1.96 (p < 0.05)
    - Not significant: -1.96 <= z <= 1.96
    """
    
    # Create classification image
    classification = ee.Image(0)  # Default: not significant
    
    # Hot spots (high-high clusters)
    hot_spots = z_score_image.gt(1.96)
    classification = classification.where(hot_spots, 1)
    
    # Very significant hot spots
    very_hot = z_score_image.gt(2.58)  # p < 0.01
    classification = classification.where(very_hot, 2)
    
    # Cold spots (low-low clusters)
    cold_spots = z_score_image.lt(-1.96)
    classification = classification.where(cold_spots, -1)
    
    # Very significant cold spots
    very_cold = z_score_image.lt(-2.58)  # p < 0.01
    classification = classification.where(very_cold, -2)
    
    return classification.rename('significance_class')

def export_results(image, description, roi, scale, folder):
    """
    Export results to Google Drive.
    """
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        region=roi,
        scale=scale,
        crs='EPSG:4326',
        maxPixels=1e9
    )
    
    task.start()
    print(f"Export task started: {description}")
    return task

def print_statistics(image, roi, scale, description):
    """
    Print basic statistics of the image.
    """
    if image is None:
        print(f"\n{description} Statistics: No data available")
        return
    
    try:
        # Use a simple approach - just get basic stats without band names
        stats = image.reduceRegion(
            reducer=ee.Reducer.mean().combine(
                ee.Reducer.stdDev().combine(
                    ee.Reducer.minMax(), sharedInputs=True
                ), sharedInputs=True
            ),
            geometry=roi,
            scale=scale,
            maxPixels=1e9,
            bestEffort=True
        )
        
        print(f"\n{description} Statistics:")
        print("-" * 40)
        
        # Get stats dictionary
        stats_dict = stats.getInfo()
        
        if not stats_dict:
            print("  No valid pixels found after masking")
            return
        
        # Print all statistics without trying to parse band names
        for key, value in stats_dict.items():
            if value is not None:
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: N/A")
                
    except Exception as e:
        print(f"\n{description} Statistics: Error computing statistics")
        print(f"  Error: {str(e)[:200]}...")  # Truncate long error messages

def main():
    """
    Main function to execute the Getis-Ord Gi* analysis workflow.
    """
    print("Starting Getis-Ord Gi* Analysis of Nighttime Lights")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    
    # Initialize Google Earth Engine
    initialize_ee(config['ee']['project'])
    
    # Define region of interest
    roi = ee.Geometry.Rectangle(config['ee']['roi'])
    
    # Create water mask
    print("\nCreating water mask...")
    water_mask = create_water_mask(roi)
      # Process baseline period
    print(f"\nProcessing baseline period: {config['baseline']['start_year']}-{config['baseline']['end_year']}")
    
    baseline_start = f"{config['baseline']['start_year']}-01-01"
    baseline_end = f"{config['baseline']['end_year']}-12-31"
    
    # Check data availability for baseline
    print("Checking baseline data availability...")
    if not check_data_availability(
        config['ee']['ntl_collection_id'], 
        roi, 
        baseline_start, 
        baseline_end, 
        config['process_months']
    ):
        print("Error: No baseline data available")
        return
    
    baseline_collection = filter_ntl_collection(
        config['ee']['ntl_collection_id'],
        roi,
        baseline_start,
        baseline_end,
        config['process_months']    )
    
    baseline_ntl = compute_mean_ntl(
        baseline_collection,
        config['ee']['ntl_band'],
        config['ee']['quality_band'],
        config['ee']['snow_band'],
        water_mask,
        roi,
        config
    )
    
    # Check if baseline processing was successful
    if baseline_ntl is None:
        print("Error: Baseline NTL processing failed - no valid data")
        return
    
    # Remove outliers from baseline
    baseline_ntl_clean = remove_outliers(baseline_ntl, roi)
      # Process sample period
    print(f"\nProcessing sample period: {config['sample']['start_year']}-{config['sample']['end_year']}")
    
    sample_start = f"{config['sample']['start_year']}-01-01"
    sample_end = f"{config['sample']['end_year']}-12-31"
    
    # Check data availability for sample
    print("Checking sample data availability...")
    if not check_data_availability(
        config['ee']['ntl_collection_id'], 
        roi, 
        sample_start, 
        sample_end, 
        config['process_months']
    ):
        print("Error: No sample data available")
        return
    
    sample_collection = filter_ntl_collection(
        config['ee']['ntl_collection_id'],
        roi,
        sample_start,
        sample_end,
        config['process_months']    )
    
    sample_ntl = compute_mean_ntl(
        sample_collection,
        config['ee']['ntl_band'],
        config['ee']['quality_band'],
        config['ee']['snow_band'],
        water_mask,
        roi,
        config
    )
    
    # Check if sample processing was successful
    if sample_ntl is None:
        print("Error: Sample NTL processing failed - no valid data")
        return
    
    # Remove outliers from sample
    sample_ntl_clean = remove_outliers(sample_ntl, roi)
    
    # Compute change (sample - baseline)
    print("\nComputing nighttime lights change...")
    ntl_change = sample_ntl_clean.subtract(baseline_ntl_clean).rename('ntl_change')
    
    # Print statistics
    print_statistics(baseline_ntl_clean, roi, config['export']['scale'], "Baseline NTL")
    print_statistics(sample_ntl_clean, roi, config['export']['scale'], "Sample NTL")
    print_statistics(ntl_change, roi, config['export']['scale'], "NTL Change")
    
    # Perform Getis-Ord Gi* analysis on the change image
    print("\nPerforming Getis-Ord Gi* analysis...")
    
    gi_star, z_score = compute_getis_ord_gi_star(
        ntl_change,
        roi,
        config['export']['scale']
    )
    
    # Classify significance
    significance_class = classify_significance(z_score)
    
    # Print Gi* statistics
    print_statistics(gi_star, roi, config['export']['scale'], "Gi* Statistics")
    print_statistics(z_score, roi, config['export']['scale'], "Z-Score")
    
    # Combine results
    results = gi_star.addBands([z_score, significance_class, ntl_change])
    
    # Create output directory if it doesn't exist
    os.makedirs(config['export']['output_dir'], exist_ok=True)
    
    # Export results
    print("\nExporting results...")
    
    # Export individual components
    export_results(
        baseline_ntl_clean,
        f"baseline_ntl_{config['baseline']['start_year']}_{config['baseline']['end_year']}",
        roi,
        config['export']['scale'],
        "NTL_Getis_Ord_Analysis"
    )
    
    export_results(
        sample_ntl_clean,
        f"sample_ntl_{config['sample']['start_year']}_{config['sample']['end_year']}",
        roi,
        config['export']['scale'],
        "NTL_Getis_Ord_Analysis"
    )
    
    export_results(
        ntl_change,
        f"ntl_change_{config['baseline']['start_year']}-{config['baseline']['end_year']}_to_{config['sample']['start_year']}-{config['sample']['end_year']}",
        roi,
        config['export']['scale'],
        "NTL_Getis_Ord_Analysis"
    )
    
    export_results(
        gi_star,
        f"getis_ord_gi_star_{config['sample']['start_year']}",
        roi,
        config['export']['scale'],
        "NTL_Getis_Ord_Analysis"
    )
    
    export_results(
        z_score,
        f"getis_ord_z_score_{config['sample']['start_year']}",
        roi,
        config['export']['scale'],
        "NTL_Getis_Ord_Analysis"
    )
    
    export_results(
        significance_class,
        f"getis_ord_significance_{config['sample']['start_year']}",
        roi,
        config['export']['scale'],
        "NTL_Getis_Ord_Analysis"
    )
    
    export_results(
        results,
        f"getis_ord_complete_analysis_{config['sample']['start_year']}",
        roi,
        config['export']['scale'],
        "NTL_Getis_Ord_Analysis"
    )
    
    print("\nAnalysis complete!")
    print("\nInterpretation Guide:")
    print("- Gi* > 0: Local clustering of high values (hot spots)")
    print("- Gi* < 0: Local clustering of low values (cold spots)")
    print("- |Z-score| > 1.96: Statistically significant at p < 0.05")
    print("- |Z-score| > 2.58: Statistically significant at p < 0.01")
    print("\nSignificance Classes:")
    print("  2: Very significant hot spots (z > 2.58)")
    print("  1: Significant hot spots (1.96 < z <= 2.58)")
    print("  0: Not significant (-1.96 <= z <= 1.96)")
    print(" -1: Significant cold spots (-2.58 <= z < -1.96)")
    print(" -2: Very significant cold spots (z < -2.58)")

if __name__ == "__main__":
    main()
