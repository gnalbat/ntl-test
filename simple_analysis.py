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

def get_individual_monthly_ntl(collection_id, roi, year, month, band_name, quality_band, snow_band, water_mask=None):
    """Get NTL data for a specific month in a specific year (no compositing across years)."""
    
    # Define start and end dates for this month in this year
    if month in [1, 3, 5, 7, 8, 10, 12]:
        days_in_month = 31
    elif month in [4, 6, 9, 11]:
        days_in_month = 30
    else:  # February
        days_in_month = 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28
        
    start_date = f"{year}-{month:02d}-01"
    end_date = f"{year}-{month:02d}-{days_in_month}"
    
    # Get images for this specific month in this specific year
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
    
    # Get collection size
    size = masked_collection.size().getInfo()
    print(f"    Found {size} images for {get_month_name(month)} {year} (after quality/snow filtering)")
    
    if size > 0:
        # Use median composite for this specific month
        composite = masked_collection.median().clip(roi)
        print(f"    ✓ Quality mask, snow mask, and water mask applied to {get_month_name(month)} {year}")
        return composite
    else:
        print(f"    Found 0 valid images for {get_month_name(month)} {year} after masking")
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

def local_getis_ord_gi_star(image, roi, scale=500, kernel_radius=7):
    """
    Enhanced Getis-Ord Gi* with local standardization for VIIRS (500m resolution).
    Kernel radius of 7 pixels = 3.5km neighborhood (optimal for urban hotspot detection).
    """
    # Define Gaussian kernel (distance-weighted, normalized)
    kernel = ee.Kernel.gaussian(
        radius=kernel_radius, 
        units='pixels', 
        normalize=True, 
        magnitude=1
    )
    
    # Calculate local mean and std dev using the same kernel
    local_mean = image.convolve(kernel)
    local_sq = image.pow(2).convolve(kernel)
    local_std = local_sq.subtract(local_mean.pow(2)).sqrt().max(0.001)  # Avoid division by zero
    
    # Calculate local Gi* (standardized by neighborhood statistics)
    gi_star = image.subtract(local_mean).divide(local_std)
    
    # Dynamic clamping based on data distribution
    def auto_clamp(gi_img):
        stats = gi_img.reduceRegion(
            reducer=ee.Reducer.percentile([1, 99]),
            geometry=roi,
            scale=scale,
            maxPixels=1e9
        ).getInfo()
        
        clamp_min = stats.get('p1', -3.5)  # Default to -3.5 if missing
        clamp_max = stats.get('p99', 3.5)  # Default to +3.5 if missing
        return gi_img.clamp(clamp_min, clamp_max)
    
    gi_star_clamped = auto_clamp(gi_star)
    
    return gi_star_clamped.rename('gi_star_local')

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
    
def robust_local_gi_star(image, roi, scale=500):
    """Local Gi* with edge correction and adaptive kernels."""
    # 1. Buffer ROI
    buffered_roi = roi.buffer(2000)  # 2km buffer
    
    # 2. Calculate urban fraction for kernel sizing
    urban_fraction = image.gt(10).reduceRegion(
        ee.Reducer.mean(), buffered_roi, scale
    ).getInfo().get('constant', 0.2)
    
    kernel_radius = 3 if urban_fraction > 0.3 else 5  # Adaptive sizing
    
    # 3. Non-normalized kernel (critical)
    kernel = ee.Kernel.square(
        radius=kernel_radius,
        units='pixels',
        normalize=False  # Weights sum to actual neighbor count
    )
    
    # 4. Local Gi* calculation
    neighbor_counts = image.unmask(0).convolve(kernel)
    local_sum = image.unmask(0).convolve(kernel)
    local_mean = local_sum.divide(neighbor_counts)
    local_var = image.pow(2).unmask(0).convolve(kernel) \
                .divide(neighbor_counts) \
                .subtract(local_mean.pow(2))
    local_std = local_var.sqrt().max(0.001)
    
    gi_star = image.subtract(local_mean).divide(local_std)
    
    # 5. Clip to original ROI
    return gi_star.clip(roi).rename('gi_star_robust')

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
    """Main function for monthly temporal analysis - processing all individual months."""
    
    print("Monthly Temporal Getis-Ord Gi* Analysis")
    print("Individual Month Processing (No Compositing)")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    
    # Initialize Google Earth Engine
    initialize_ee(config['ee']['project'])
    
    # Define region of interest
    roi = ee.Geometry.Rectangle(config['ee']['roi'])
    
    print(f"\nProcessing region: {config['ee']['roi']}")
    print(f"Output directory: {config['export']['output_dir']}")
    print(f"Sample period: {config['sample']['start_year']}-{config['sample']['end_year']}")
    print(f"Baseline period: {config['baseline']['start_year']}-{config['baseline']['end_year']}")
    
    # Create water mask
    print("\nCreating water mask...")
    water_mask = create_water_mask(roi)
    
    # Generate list of all months to process
    months_to_process = config.get('process_months', list(range(1, 13)))
    years_to_process = list(range(config['sample']['start_year'], config['sample']['end_year'] + 1))
    
    total_combinations = len(years_to_process) * len(months_to_process)
    current_combination = 0
    
    print(f"\n📅 Processing Plan:")
    print(f"   Years: {config['sample']['start_year']} to {config['sample']['end_year']} ({len(years_to_process)} years)")
    print(f"   Months: {len(months_to_process)} months per year")
    print(f"   Total individual months: {total_combinations}")
    print(f"   Each month compared against baseline composite ({config['baseline']['start_year']}-{config['baseline']['end_year']})")
    
    # Create baseline composites for each month (once per month across all baseline years)
    baseline_composites = {}
    print(f"\n🏗️ Creating baseline composites...")
    
    for month in months_to_process:
        month_name = get_month_name(month)
        print(f"\n  Creating baseline composite for {month_name} ({config['baseline']['start_year']}-{config['baseline']['end_year']})...")
        
        try:
            baseline_composite = get_monthly_ntl_composite(
                config['ee']['ntl_collection_id'],
                roi,
                config['baseline']['start_year'],
                config['baseline']['end_year'],
                month,
                config['ee']['ntl_band'],
                config['ee']['quality_band'],
                config['ee']['snow_band'],
                water_mask
            )
            
            if baseline_composite is not None:
                baseline_composites[month] = baseline_composite
                print(f"    ✓ Baseline composite created for {month_name}")
                
                # Export baseline composite (once per month)
                baseline_filename = f"baseline_{month:02d}_{month_name}_{config['baseline']['start_year']}-{config['baseline']['end_year']}"
                export_image_locally(
                    baseline_composite,
                    baseline_filename,
                    roi,
                    config['export']['scale'],
                    config['export']['output_dir']
                )
            else:
                print(f"    ✗ No baseline data for {month_name}")
                
        except Exception as e:
            print(f"    ✗ Error creating baseline for {month_name}: {e}")
    
    print(f"\n✅ Baseline composites ready: {len(baseline_composites)}/{len(months_to_process)} months")
    
    # Process each individual month for each year
    print(f"\n🔄 Processing individual months...")
    
    for year in years_to_process:
        print(f"\n{'='*80}")
        print(f"PROCESSING YEAR {year}")
        print(f"{'='*80}")
        
        for month in months_to_process:
            current_combination += 1
            month_name = get_month_name(month)
            
            print(f"\n📅 Processing {month_name} {year} ({current_combination}/{total_combinations})")
            print("-" * 50)
            
            # Skip if no baseline for this month
            if month not in baseline_composites:
                print(f"  ⏭️  Skipping - no baseline composite for {month_name}")
                continue
                
            try:
                # Get individual monthly data (no compositing)
                print(f"  📡 Fetching {month_name} {year} data...")
                sample_composite = get_individual_monthly_ntl(
                    config['ee']['ntl_collection_id'],
                    roi,
                    year,
                    month,
                    config['ee']['ntl_band'],
                    config['ee']['quality_band'],
                    config['ee']['snow_band'],
                    water_mask
                )
                
                if sample_composite is None:
                    print(f"  ✗ No sample data for {month_name} {year}, skipping...")
                    continue
                
                # Calculate change against baseline
                print(f"  🧮 Calculating NTL change for {month_name} {year}...")
                baseline_composite = baseline_composites[month]
                ntl_change = sample_composite.subtract(baseline_composite).rename('ntl_change')
                
                # Perform Gi* analysis
                print(f"  🎯 Performing Gi* analysis for {month_name} {year}...")
                gi_star = robust_local_gi_star(ntl_change, roi, config['export']['scale'])
                
                # Calculate statistics
                print(f"  📊 Computing statistics for {month_name} {year}...")
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
                    
                    print(f"    {month_name} {year} - NTL Change:")
                    for key, value in change_stats.items():
                        if value is not None:
                            print(f"      {key}: {value:.4f}")
                    
                    print(f"    {month_name} {year} - Gi* Statistics:")
                    for key, value in gi_stats.items():
                        if value is not None:
                            print(f"      {key}: {value:.4f}")
                    
                except Exception as e:
                    print(f"    ⚠️  Could not compute statistics: {e}")
                
                # Export individual month files
                print(f"  💾 Exporting {month_name} {year} results...")
                
                month_str = f"{month:02d}"
                
                # Export sample (individual month)
                sample_filename = f"sample_{month_str}_{month_name}_{year}"
                export_image_locally(
                    sample_composite,
                    sample_filename,
                    roi,
                    config['export']['scale'],
                    config['export']['output_dir']
                )
                
                # Export change
                change_filename = f"change_{month_str}_{month_name}_{year}_vs_baseline_{config['baseline']['start_year']}-{config['baseline']['end_year']}"
                export_image_locally(
                    ntl_change,
                    change_filename,
                    roi,
                    config['export']['scale'],
                    config['export']['output_dir']
                )
                
                # Export Gi*
                if gi_star is not None:
                    # Detailed analysis
                    analyze_gi_star_distribution(gi_star, roi, config['export']['scale'], f"{month_name} {year}")
                    
                    # Export Gi*
                    gi_filename = f"gi_star_{month:02d}_{month_name}_{year}"
                    export_image_locally(gi_star, gi_filename, roi, config['export']['scale'], config['export']['output_dir'])
                    print(f"      ✓ Exported: {gi_filename}")
                else:
                    print(f"      ❌ Failed to compute Gi* for {month_name} {year}")
                
                print(f"  ✅ Completed {month_name} {year}")
                
            except Exception as e:
                print(f"  ❌ Error processing {month_name} {year}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Final summary
    print(f"\n{'='*80}")
    print("MONTHLY TEMPORAL ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print(f"📁 Results saved to: {config['export']['output_dir']}")
    print(f"📅 Processed: {len(years_to_process)} years × {len(months_to_process)} months = {total_combinations} individual months")
    print(f"🏗️ Created: {len(baseline_composites)} baseline composites")
    
    print(f"\n📋 File Structure Generated:")
    print(f"   Baseline composites (one per month):")
    print(f"     baseline_MM_MonthName_YYYY-YYYY.tif")
    print(f"   ")
    print(f"   Individual monthly files (for each year):")
    for year in years_to_process:
        print(f"     {year}:")
        print(f"       sample_MM_MonthName_{year}.tif")
        print(f"       change_MM_MonthName_{year}_vs_baseline_YYYY-YYYY.tif")
        print(f"       gi_star_MM_MonthName_{year}.tif")
    
    print(f"\n🎯 Analysis Capabilities Unlocked:")
    print(f"   • Seasonal patterns: Compare same months across years")
    print(f"   • Annual trends: Track changes year-over-year")
    print(f"   • Monthly variations: Compare different months within years")
    print(f"   • Temporal clustering: Identify when and where hot/cold spots occur")
    print(f"   • Anomaly detection: Find unusual months compared to baseline")
    
    print(f"\n📊 Recommended Next Steps:")
    print(f"   1. Run visualize_results.py for spatial visualizations")
    print(f"   2. Run mann_kendall_analysis_fixed.py for temporal trend analysis")
    print(f"   3. Use the individual monthly files for detailed temporal analysis")

if __name__ == "__main__":
    main()
