#!/usr/bin/env python3
"""
Monthly Temporal Getis-Ord Gi* Analysis Script
This version processes data monthly for temporal analysis and saves locally
"""

import ee
import yaml
import os
from datetime import datetime

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def initialize_ee(project_id):
    """Initialize Google Earth Engine."""
    ee.Initialize(project=project_id)
    print(f"Google Earth Engine initialized for project: {project_id}")

def get_monthly_ntl_composite(collection_id, roi, start_year, end_year, month, band_name, quality_band, snow_band, water_mask=None):
    """Get monthly NTL composite for specified years and month with comprehensive masking."""
    
    # Create list of all years to process
    years = list(range(start_year, end_year + 1))
    
    # Create a collection for this specific month across all years
    monthly_images = []
    
    for year in years:
        # Define start and end dates for this month in this year
        if month in [1, 3, 5, 7, 8, 10, 12]:
            days_in_month = 31
        elif month in [4, 6, 9, 11]:
            days_in_month = 30
        else:  # February
            days_in_month = 28  # Simplified - not accounting for leap years
            
        start_date = f"{year}-{month:02d}-01"
        end_date = f"{year}-{month:02d}-{days_in_month}"
        
        # Get images for this month in this year
        monthly_collection = ee.ImageCollection(collection_id) \
            .filterDate(start_date, end_date) \
            .filterBounds(roi)
        
        # Apply quality and snow masking to each image
        def apply_masks(image):
            # Select the NTL band
            ntl = image.select(band_name)
            
            # Quality mask: keep only good quality pixels (≤1)
            quality_mask = image.select(quality_band).lte(1)
            
            # Snow mask: exclude snow-covered pixels (≠0)
            snow_mask = image.select(snow_band).eq(0)
            
            # Combine all masks
            combined_mask = quality_mask.And(snow_mask)
            
            # Apply water mask if provided
            if water_mask is not None:
                combined_mask = combined_mask.And(water_mask)
            
            # Apply combined mask to NTL data
            return ntl.updateMask(combined_mask)
        
        # Apply masking to the collection
        masked_collection = monthly_collection.map(apply_masks)
        
        # Add to list if collection has images
        size = masked_collection.size()
        if size.getInfo() > 0:
            monthly_images.append(masked_collection)
    
    # Combine all monthly collections
    if monthly_images:
        # Merge all collections for this month across years
        combined_collection = monthly_images[0]
        for i in range(1, len(monthly_images)):
            combined_collection = combined_collection.merge(monthly_images[i])
        
        total_images = combined_collection.size().getInfo()
        print(f"    Found {total_images} images for month {month} (after quality/snow filtering)")
        
        # Use median composite for robustness
        composite = combined_collection.median().clip(roi)
        
        print(f"    ✓ Quality mask, snow mask, and water mask applied to {get_month_name(month)} composite")
        
        return composite
    else:
        print(f"    Found 0 valid images for month {month} after masking")
        return None

def simple_getis_ord_gi_star(image, roi, scale=500):
    """Simplified Getis-Ord Gi* calculation with reliable statistics."""
    
    # Define a 3x3 kernel (not normalized for proper Gi* calculation)
    kernel = ee.Kernel.square(radius=1, units='pixels', normalize=False)
    
    # Calculate local sum
    local_sum = image.convolve(kernel)
    
    # Calculate global statistics separately for reliability
    global_mean = image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=roi,
        scale=scale,
        maxPixels=1e9,
        bestEffort=True
    ).values().get(0)
    
    global_std = image.reduceRegion(
        reducer=ee.Reducer.stdDev(),
        geometry=roi,
        scale=scale,
        maxPixels=1e9,
        bestEffort=True
    ).values().get(0)
    
    # Convert to ee.Number
    global_mean = ee.Number(global_mean)
    global_std = ee.Number(global_std)
    
    # Number of neighbors in 3x3 kernel
    n = ee.Number(9)
    
    # Calculate expected value and standard deviation
    expected = global_mean.multiply(n)
    std_dev = global_std.multiply(n.sqrt()).max(0.001)  # Prevent division by zero
    
    # Calculate unclamped Gi* statistic
    gi_star_unclamped = local_sum.subtract(expected).divide(std_dev)
    
    # Get statistics with separate calls for reliability
    print("    Analyzing unclamped Gi* statistics...")
    try:
        # Get each statistic separately
        min_result = gi_star_unclamped.reduceRegion(
            reducer=ee.Reducer.min(),
            geometry=roi,
            scale=scale,
            maxPixels=1e9,
            bestEffort=True
        ).getInfo()
        
        max_result = gi_star_unclamped.reduceRegion(
            reducer=ee.Reducer.max(),
            geometry=roi,
            scale=scale,
            maxPixels=1e9,
            bestEffort=True
        ).getInfo()
        
        mean_result = gi_star_unclamped.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=roi,
            scale=scale,
            maxPixels=1e9,
            bestEffort=True
        ).getInfo()
        
        std_result = gi_star_unclamped.reduceRegion(
            reducer=ee.Reducer.stdDev(),
            geometry=roi,
            scale=scale,
            maxPixels=1e9,
            bestEffort=True
        ).getInfo()
        
        count_result = gi_star_unclamped.reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=roi,
            scale=scale,
            maxPixels=1e9,
            bestEffort=True
        ).getInfo()
        
        # Extract values (should be single values now)
        min_val = list(min_result.values())[0] if min_result else None
        max_val = list(max_result.values())[0] if max_result else None
        mean_val = list(mean_result.values())[0] if mean_result else None
        std_val = list(std_result.values())[0] if std_result else None
        count_val = int(list(count_result.values())[0]) if count_result else 0
        
        print(f"    📊 Unclamped Gi* Statistics:")
        print(f"       Range: {min_val:.3f} to {max_val:.3f}")
        print(f"       Mean: {mean_val:.3f}, Std: {std_val:.3f}")
        print(f"       Valid pixels: {count_val}")
        
        # Calculate clamping impact
        if count_val > 0 and min_val is not None and max_val is not None:
            extreme_low_count = 0
            extreme_high_count = 0
            
            # Count pixels that will be clamped
            if min_val < -5:
                extreme_low_result = gi_star_unclamped.lt(-5).reduceRegion(
                    reducer=ee.Reducer.sum(),
                    geometry=roi,
                    scale=scale,
                    maxPixels=1e9,
                    bestEffort=True
                ).getInfo()
                extreme_low_count = int(list(extreme_low_result.values())[0] or 0)
            
            if max_val > 5:
                extreme_high_result = gi_star_unclamped.gt(5).reduceRegion(
                    reducer=ee.Reducer.sum(),
                    geometry=roi,
                    scale=scale,
                    maxPixels=1e9,
                    bestEffort=True
                ).getInfo()
                extreme_high_count = int(list(extreme_high_result.values())[0] or 0)
            
            total_extreme = extreme_low_count + extreme_high_count
            extreme_percent = (total_extreme / count_val) * 100
            
            print(f"    🔒 Clamping Impact (sets floor/ceiling, doesn't remove pixels):")
            print(f"       Pixels < -5 (will be set to -5): {extreme_low_count} ({(extreme_low_count/count_val)*100:.2f}%)")
            print(f"       Pixels > +5 (will be set to +5): {extreme_high_count} ({(extreme_high_count/count_val)*100:.2f}%)")
            print(f"       Total affected: {total_extreme} ({extreme_percent:.2f}%)")
            
            # Provide recommendations
            if extreme_percent > 5:
                print(f"    ⚠️  WARNING: {extreme_percent:.1f}% of pixels affected by clamping!")
            elif extreme_percent > 1:
                print(f"    ⚡ CAUTION: {extreme_percent:.1f}% of pixels affected by clamping")
            else:
                print(f"    ✅ GOOD: Only {extreme_percent:.1f}% of pixels affected by clamping")
        
    except Exception as e:
        print(f"    ❌ Error calculating unclamped statistics: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Apply clamping (sets floor and ceiling, preserves all pixels)
    clamp_min, clamp_max = -5, 5
    gi_star_clamped = gi_star_unclamped.clamp(clamp_min, clamp_max)
    
    print(f"    🔒 Applied clamping: [{clamp_min}, {clamp_max}] (floor/ceiling, all pixels preserved)")
    
    # Get final clamped statistics
    try:
        min_clamped = gi_star_clamped.reduceRegion(
            reducer=ee.Reducer.min(),
            geometry=roi,
            scale=scale,
            maxPixels=1e9,
            bestEffort=True
        ).getInfo()
        
        max_clamped = gi_star_clamped.reduceRegion(
            reducer=ee.Reducer.max(),
            geometry=roi,
            scale=scale,
            maxPixels=1e9,
            bestEffort=True
        ).getInfo()
        
        mean_clamped = gi_star_clamped.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=roi,
            scale=scale,
            maxPixels=1e9,
            bestEffort=True
        ).getInfo()
        
        clamped_min = list(min_clamped.values())[0] if min_clamped else None
        clamped_max = list(max_clamped.values())[0] if max_clamped else None
        clamped_mean = list(mean_clamped.values())[0] if mean_clamped else None
        
        print(f"    📊 Final Gi* Statistics (after clamping):")
        print(f"       Range: {clamped_min:.3f} to {clamped_max:.3f}")
        print(f"       Mean: {clamped_mean:.3f}")
        
    except Exception as e:
        print(f"    ❌ Error calculating clamped statistics: {str(e)}")
    
    return gi_star_clamped.rename('gi_star')

def export_image_locally(image, filename, roi, scale=500, output_dir="./data"):
    """Export image as GeoTIFF to local folder."""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Full file path
    filepath = os.path.join(output_dir, f"{filename}.tif")
    
    # Get download URL
    url = image.getDownloadURL({
        'scale': scale,
        'crs': 'EPSG:4326',
        'region': roi,
        'format': 'GEO_TIFF'
    })
    
    print(f"    Downloading: {filename}.tif")
    
    # Download the file
    import urllib.request
    try:
        urllib.request.urlretrieve(url, filepath)
        print(f"    ✓ Saved: {filepath}")
        return True
    except Exception as e:
        print(f"    ✗ Error downloading {filename}: {e}")
        return False

def get_month_name(month_num):
    """Convert month number to month name."""
    months = ['', 'January', 'February', 'March', 'April', 'May', 'June',
              'July', 'August', 'September', 'October', 'November', 'December']
    return months[month_num]

def create_water_mask(roi):
    """
    Create a water mask using JRC Global Surface Water dataset
    to exclude water bodies from analysis.
    """
    try:
        # JRC Global Surface Water dataset
        gsw = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
        
        # Create water mask (1 for land, 0 for water)
        # Use conservative threshold - only exclude areas that are water >90% of the time
        water_mask = gsw.select('occurrence').lt(90).clip(roi)
        
        print("    ✓ Water mask created successfully")
        return water_mask
        
    except Exception as e:
        print(f"    Warning: Could not create water mask - {str(e)[:100]}...")
        print("    Proceeding without water masking")
        # Return a mask that doesn't filter anything
        return ee.Image.constant(1).clip(roi)

def analyze_gi_star_distribution(gi_star_image, roi, scale=500, month_name=""):
    """Analyze the distribution of Gi* values and provide insights."""
    
    print(f"\n📈 Detailed Gi* Distribution Analysis for {month_name}:")
    print("=" * 60)
    
    try:
        # Simple statistics first
        simple_stats = gi_star_image.reduceRegion(
            reducer=ee.Reducer.minMax().combine(
                reducer2=ee.Reducer.mean().combine(
                    reducer2=ee.Reducer.count(),
                    sharedInputs=True
                ),
                sharedInputs=True
            ),
            geometry=roi,
            scale=scale,
            maxPixels=1e9,
            bestEffort=True
        )
        
        stats_info = simple_stats.getInfo()
        print(f"Debug - Raw stats info: {stats_info}")
        
        if stats_info and len(stats_info) > 0:
            # Extract values safely
            values = list(stats_info.values())
            if len(values) >= 4:  # min, max, mean, count
                min_val = values[0] if values[0] is not None else 0
                max_val = values[1] if values[1] is not None else 0
                mean_val = values[2] if values[2] is not None else 0
                count_val = values[3] if values[3] is not None else 0
                
                print(f"📊 Distribution Summary:")
                print(f"   Range: {min_val:.3f} to {max_val:.3f}")
                print(f"   Mean: {mean_val:.3f}")
                print(f"   Valid pixels: {count_val}")
                
                # Statistical significance analysis
                print(f"\n🎯 Statistical Significance Analysis:")
                sig_001 = abs(min_val) > 3.29 or abs(max_val) > 3.29
                sig_01 = abs(min_val) > 2.58 or abs(max_val) > 2.58
                sig_05 = abs(min_val) > 1.96 or abs(max_val) > 1.96
                
                print(f"   99.9% confidence (|Gi*| > 3.29): {'Yes' if sig_001 else 'No'}")
                print(f"   99% confidence   (|Gi*| > 2.58): {'Yes' if sig_01 else 'No'}")
                print(f"   95% confidence   (|Gi*| > 1.96): {'Yes' if sig_05 else 'No'}")
                
                # Clustering interpretation
                print(f"\n🗺️  Spatial Clustering Interpretation:")
                if mean_val > 0.5:
                    print(f"   Strong positive clustering (hot spots dominate)")
                elif mean_val > 0.1:
                    print(f"   Moderate positive clustering")
                elif mean_val < -0.5:
                    print(f"   Strong negative clustering (cold spots dominate)")
                elif mean_val < -0.1:
                    print(f"   Moderate negative clustering")
                else:
                    print(f"   Weak clustering (near random distribution)")
                
                if count_val == 0:
                    print(f"   ⚠️  WARNING: No valid pixels in Gi* result!")
                    print(f"   This might indicate issues with:")
                    print(f"   - Masking (too aggressive)")
                    print(f"   - Edge effects (kernel convolution)")
                    print(f"   - Data quality (insufficient valid data)")
            else:
                print(f"❌ Insufficient statistics returned: {len(values)} values")
        else:
            print(f"❌ No statistics returned from Gi* image")
            
    except Exception as e:
        print(f"❌ Error in distribution analysis: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """Main function for monthly temporal analysis."""
    
    print("Monthly Temporal Getis-Ord Gi* Analysis")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    
    # Initialize Google Earth Engine
    initialize_ee(config['ee']['project'])
      # Define region of interest
    roi = ee.Geometry.Rectangle(config['ee']['roi'])
    
    print(f"\nProcessing region: {config['ee']['roi']}")
    print(f"Output directory: {config['export']['output_dir']}")
    
    # Create water mask
    print("\nCreating water mask...")
    water_mask = create_water_mask(roi)
    
    # Process each month
    months_to_process = config.get('process_months', list(range(1, 13)))
    
    for month in months_to_process:
        month_name = get_month_name(month)
        print(f"\n{'='*60}")
        print(f"Processing {month_name} (Month {month})")        # Get baseline composite for this month
        print(f"\n  Creating baseline composite for {month_name} ({config['baseline']['start_year']}-{config['baseline']['end_year']})...")
        
        try:
            baseline_composite = get_monthly_ntl_composite(
                config['ee']['ntl_collection_id'],
                roi,
                config['baseline']['start_year'],
                config['baseline']['end_year'],
                month,
                config['ee']['ntl_band'],
                config['ee']['quality_band'],  # Add quality band
                config['ee']['snow_band'],     # Add snow band
                water_mask
            )
            
            if baseline_composite is None:
                print(f"  ✗ No baseline data for {month_name}, skipping...")
                continue
              # Get sample composite for this month
            print(f"  Creating sample composite for {month_name} ({config['sample']['start_year']}-{config['sample']['end_year']})...")
            
            sample_composite = get_monthly_ntl_composite(
                config['ee']['ntl_collection_id'],
                roi,
                config['sample']['start_year'],
                config['sample']['end_year'],
                month,
                config['ee']['ntl_band'],
                config['ee']['quality_band'],  # Add quality band
                config['ee']['snow_band'],     # Add snow band
                water_mask
            )
            
            if sample_composite is None:
                print(f"  ✗ No sample data for {month_name}, skipping...")
                continue
            
            # Calculate change
            print(f"  Calculating NTL change for {month_name}...")
            ntl_change = sample_composite.subtract(baseline_composite).rename('ntl_change')
            
            # Perform Gi* analysis
            print(f"  Performing Gi* analysis for {month_name}...")
            gi_star = simple_getis_ord_gi_star(ntl_change, roi, config['export']['scale'])
            
            # Calculate basic statistics
            print(f"  Computing statistics for {month_name}...")
            try:
                # Change statistics
                change_stats = ntl_change.reduceRegion(
                    reducer=ee.Reducer.mean().combine(ee.Reducer.minMax(), sharedInputs=True),
                    geometry=roi,
                    scale=config['export']['scale'],
                    maxPixels=1e9,
                    bestEffort=True
                ).getInfo()
                
                # Gi* statistics
                gi_stats = gi_star.reduceRegion(
                    reducer=ee.Reducer.mean().combine(ee.Reducer.minMax(), sharedInputs=True),
                    geometry=roi,
                    scale=config['export']['scale'],
                    maxPixels=1e9,
                    bestEffort=True
                ).getInfo()
                
                print(f"  {month_name} - NTL Change Statistics:")
                for key, value in change_stats.items():
                    if value is not None:
                        print(f"    {key}: {value:.4f}")
                
                print(f"  {month_name} - Gi* Statistics:")
                for key, value in gi_stats.items():
                    if value is not None:
                        print(f"    {key}: {value:.4f}")
                
            except Exception as e:
                print(f"  Warning: Could not compute statistics for {month_name}: {e}")
            
            # Export files locally
            print(f"  Exporting {month_name} results...")
            
            month_str = f"{month:02d}"
            
            # Export baseline
            export_image_locally(
                baseline_composite,
                f"baseline_{month_str}_{month_name}_{config['baseline']['start_year']}-{config['baseline']['end_year']}",
                roi,
                config['export']['scale'],
                config['export']['output_dir']
            )
            
            # Export sample
            export_image_locally(
                sample_composite,
                f"sample_{month_str}_{month_name}_{config['sample']['start_year']}-{config['sample']['end_year']}",
                roi,
                config['export']['scale'],
                config['export']['output_dir']
            )
            
            # Export change
            export_image_locally(
                ntl_change,
                f"change_{month_str}_{month_name}_{config['baseline']['start_year']}-{config['baseline']['end_year']}_to_{config['sample']['start_year']}-{config['sample']['end_year']}",
                roi,
                config['export']['scale'],
                config['export']['output_dir']
            )
            
            # Export Gi*
            if gi_star is not None:
                # Detailed analysis
                analyze_gi_star_distribution(gi_star, roi, config['export']['scale'], month_name)
                
                # Export as before
                gi_filename = f"gi_star_{month:02d}_{month_name}_{config['sample']['start_year']}"
                export_image_locally(gi_star, gi_filename, roi, config['export']['scale'], config['export']['output_dir'])
                print(f"    ✓ Exported: {gi_filename}")
            else:
                print(f"    ❌ Failed to compute Gi* for {month_name}")
            
            print(f"  ✓ Completed processing for {month_name}")
            
        except Exception as e:
            print(f"  ✗ Error processing {month_name}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("Monthly Temporal Analysis Complete!")
    print(f"{'='*60}")
    print(f"Results saved to: {config['export']['output_dir']}")
    print("\nFiles generated for each month:")
    print("  - baseline_MM_MonthName_YYYY-YYYY.tif")
    print("  - sample_MM_MonthName_YYYY-YYYY.tif") 
    print("  - change_MM_MonthName_YYYY-YYYY_to_YYYY-YYYY.tif")
    print("  - gi_star_MM_MonthName_YYYY.tif")
    print("\nUse these files for temporal analysis to identify:")
    print("  • Seasonal patterns in nighttime lights")
    print("  • Monthly variations in spatial clustering")
    print("  • Temporal trends in hot/cold spots")

if __name__ == "__main__":
    main()
