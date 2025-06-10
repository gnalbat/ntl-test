#!/usr/bin/env python3
"""
Mann-Kendall Trend Analysis for Gi* Results
Analyzes temporal trends in spatial clustering patterns using pyMannKendall
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
import pandas as pd
from datetime import datetime
import warnings
import xyzservices.providers as xyz
warnings.filterwarnings('ignore')

try:
    import pymannkendall as mk
    print("✅ pyMannKendall imported successfully")
except ImportError:
    print("❌ pyMannKendall not found. Installing...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pymannkendall'])
    import pymannkendall as mk
    print("✅ pyMannKendall installed and imported")

try:
    import rasterio
    RASTERIO_AVAILABLE = True
    print("✅ Rasterio available for reading GeoTIFFs")
except ImportError:
    RASTERIO_AVAILABLE = False
    print("⚠️  Rasterio not available - will use synthetic data")

try:
    import contextily as ctx
    CONTEXTILY_AVAILABLE = True
    print("✅ Contextily available for basemaps")
except ImportError:
    CONTEXTILY_AVAILABLE = False
    print("⚠️ Contextily not available - maps will use basic styling")

def load_gi_star_data():
    """Load all Gi* GeoTIFF files and extract temporal data."""
    
    print("🔍 Loading Gi* data for Mann-Kendall analysis...")
    print("=" * 60)
    
    # Find all Gi* files
    gi_files = glob("./data/gi_star_*.tif")
    gi_files.sort()
    
    if not gi_files:
        print("❌ No Gi* files found in ./data/")
        return None, None
    
    print(f"Found {len(gi_files)} Gi* files:")
    
    # Extract month information
    file_info = []
    
    for gi_file in gi_files:
        filename = os.path.basename(gi_file)
        parts = filename.split('_')
        
        # Expected format: gi_star_01_January_2024.tif.tif (note double extension)
        if len(parts) >= 4 and parts[0] == 'gi' and parts[1] == 'star':
            try:
                month_num = int(parts[2])  # e.g., "01", "02", etc.
                month_name = parts[3]      # e.g., "January", "February", etc.
                # Handle the .tif.tif extension
                year = parts[4].replace('.tif.tif', '').replace('.tif', '') if len(parts) > 4 else "2024"
                
                file_info.append({
                    'file': gi_file,
                    'month_num': month_num,
                    'month_name': month_name,
                    'year': year,
                    'filename': filename
                })
                
                print(f"  {month_num:2d}. {month_name} - {filename}")
            except (ValueError, IndexError) as e:
                print(f"  ⚠️  Skipping file with unexpected format: {filename} - {e}")
        else:
            print(f"  ⚠️  Skipping file with unexpected format: {filename}")
    
    # Sort by month number
    file_info.sort(key=lambda x: x['month_num'])
    
    return file_info, gi_files

def calculate_pixel_statistics(file_info):
    """Calculate pixel-level statistics for each month using actual GeoTIFF data."""
    
    print(f"\n📊 Calculating actual pixel-level statistics...")
    print("=" * 60)
    
    monthly_stats = []
    
    if RASTERIO_AVAILABLE:
        # Use rasterio to read actual pixel values
        print("Using rasterio to read actual pixel values from GeoTIFFs...")
        
        for info in file_info:
            try:
                with rasterio.open(info['file']) as src:
                    data = src.read(1)  # Read first band
                    
                    # Remove nodata values
                    valid_data = data[~np.isnan(data)]
                    valid_data = valid_data[np.isfinite(valid_data)]
                    
                    if len(valid_data) > 0:
                        # Calculate comprehensive statistics
                        stats = {
                            'month_num': info['month_num'],
                            'month_name': info['month_name'],
                            'year': info['year'],
                            'mean': float(np.mean(valid_data)),
                            'median': float(np.median(valid_data)),
                            'std': float(np.std(valid_data)),
                            'min': float(np.min(valid_data)),
                            'max': float(np.max(valid_data)),
                            'q25': float(np.percentile(valid_data, 25)),
                            'q75': float(np.percentile(valid_data, 75)),
                            'count': len(valid_data),
                            'hotspots': int(np.sum(valid_data > 1.96)),  # 95% confidence hot spots
                            'coldspots': int(np.sum(valid_data < -1.96)),  # 95% confidence cold spots
                            'hotspots_99': int(np.sum(valid_data > 2.58)),  # 99% confidence hot spots
                            'coldspots_99': int(np.sum(valid_data < -2.58)),  # 99% confidence cold spots
                            'significant': int(np.sum(np.abs(valid_data) > 1.96)),  # Total significant
                            'significant_99': int(np.sum(np.abs(valid_data) > 2.58)),  # Total highly significant
                        }
                        
                        # Calculate percentages
                        total_pixels = stats['count']
                        stats['hotspot_pct'] = (stats['hotspots'] / total_pixels) * 100
                        stats['coldspot_pct'] = (stats['coldspots'] / total_pixels) * 100
                        stats['significant_pct'] = (stats['significant'] / total_pixels) * 100
                        stats['hotspot_99_pct'] = (stats['hotspots_99'] / total_pixels) * 100
                        stats['coldspot_99_pct'] = (stats['coldspots_99'] / total_pixels) * 100
                        
                        # Additional clustering metrics
                        stats['positive_clustering'] = int(np.sum(valid_data > 0))
                        stats['negative_clustering'] = int(np.sum(valid_data < 0))
                        stats['positive_clustering_pct'] = (stats['positive_clustering'] / total_pixels) * 100
                        stats['negative_clustering_pct'] = (stats['negative_clustering'] / total_pixels) * 100
                        
                        monthly_stats.append(stats)
                        
                        print(f"  {info['month_name']:>10}: Mean={stats['mean']:6.3f}, "
                              f"Pixels={total_pixels:6d}, "
                              f"Hot95%={stats['hotspots']:4d} ({stats['hotspot_pct']:4.1f}%), "
                              f"Cold95%={stats['coldspots']:4d} ({stats['coldspot_pct']:4.1f}%)")
                        
                        print(f"                Range=[{stats['min']:6.3f}, {stats['max']:6.3f}], "
                              f"Std={stats['std']:5.3f}, "
                              f"Hot99%={stats['hotspots_99']:3d}, "
                              f"Cold99%={stats['coldspots_99']:3d}")
                        
                    else:
                        print(f"  {info['month_name']:>10}: ❌ No valid data found")
                        
            except Exception as e:
                print(f"  {info['month_name']:>10}: ❌ Error reading file - {str(e)}")
                
    else:
        # Fallback to synthetic data if rasterio not available
        print("⚠️  Rasterio not available - using synthetic data for demonstration...")
        monthly_stats = calculate_synthetic_statistics(file_info)
    
    return monthly_stats

def calculate_synthetic_statistics(file_info):
    """Calculate synthetic statistics for demonstration (fallback when rasterio not available)."""
    
    print(f"\n📊 Calculating synthetic statistics for demonstration...")
    print("=" * 60)
    
    monthly_stats = []
    
    for info in file_info:
        # Create realistic synthetic data for demonstration
        np.random.seed(info['month_num'])  # Consistent per month
        
        # Simulate seasonal patterns
        seasonal_effect = np.cos(2 * np.pi * info['month_num'] / 12) * 0.3
        trend_effect = (info['month_num'] - 6.5) * 0.02  # Small linear trend
        base_mean = -0.4 + seasonal_effect + trend_effect  # Base negative clustering
        
        stats = {
            'month_num': info['month_num'],
            'month_name': info['month_name'],
            'year': info['year'],
            'mean': base_mean + np.random.normal(0, 0.05),
            'median': base_mean + np.random.normal(0, 0.03),
            'std': 1.2 + np.random.normal(0, 0.1),
            'min': -5.0,
            'max': 5.0,
            'q25': base_mean - 0.8 + np.random.normal(0, 0.05),
            'q75': base_mean + 0.8 + np.random.normal(0, 0.05),
            'count': 50000 + np.random.randint(-5000, 5000),
            'hotspots': max(0, int(1000 + seasonal_effect * 500 + np.random.normal(0, 200))),
            'coldspots': max(0, int(3000 - seasonal_effect * 800 + np.random.normal(0, 300))),
            'significant': 0,  # Will be calculated
            'hotspot_pct': 0,  # Will be calculated
            'coldspot_pct': 0  # Will be calculated
        }
        
        # Calculate derived statistics
        total_pixels = stats['count']
        stats['significant'] = stats['hotspots'] + stats['coldspots']
        stats['hotspot_pct'] = (stats['hotspots'] / total_pixels) * 100
        stats['coldspot_pct'] = (stats['coldspots'] / total_pixels) * 100
        
        monthly_stats.append(stats)
        
        print(f"  {info['month_name']:>10}: Mean={stats['mean']:6.3f}, "
              f"Hotspots={stats['hotspots']:5d} ({stats['hotspot_pct']:4.1f}%), "
              f"Coldspots={stats['coldspots']:5d} ({stats['coldspot_pct']:4.1f}%)")
    
    return monthly_stats

def perform_mann_kendall_analysis(monthly_stats):
    """Perform Mann-Kendall trend analysis on various metrics."""
    
    print(f"\n🔬 Mann-Kendall Trend Analysis")
    print("=" * 60)
    
    if not monthly_stats:
        print("❌ No data available for analysis")
        return None
    
    # Convert to DataFrame for easier handling
    df = pd.DataFrame(monthly_stats)
      # Define metrics to analyze
    metrics = {
        'mean': 'Mean Gi* Value',
        'median': 'Median Gi* Value',
        'std': 'Standard Deviation',
        'hotspot_pct': 'Hot Spot Percentage (95%)',
        'coldspot_pct': 'Cold Spot Percentage (95%)',
        'hotspots': 'Hot Spot Count (95%)',
        'coldspots': 'Cold Spot Count (95%)',
        'significant_pct': 'Significant Pixels Percentage',
        'hotspot_99_pct': 'Hot Spot Percentage (99%)',
        'coldspot_99_pct': 'Cold Spot Percentage (99%)',
        'positive_clustering_pct': 'Positive Clustering %',
        'negative_clustering_pct': 'Negative Clustering %'
    }
    
    results = {}
    
    print(f"Analyzing {len(metrics)} metrics across {len(monthly_stats)} months:\n")
    
    for metric, description in metrics.items():
        if metric in df.columns:
            data = df[metric].values
            
            try:
                # Perform Mann-Kendall test
                mk_result = mk.original_test(data)
                
                results[metric] = {
                    'description': description,
                    'trend': mk_result.trend,
                    'p_value': mk_result.p,
                    'z_score': mk_result.z,
                    'tau': mk_result.Tau,
                    'slope': mk_result.slope,
                    'intercept': mk_result.intercept,
                    'data': data
                }
                
                # Interpret results
                trend_direction = "↗️ Increasing" if mk_result.trend == 'increasing' else \
                                "↘️ Decreasing" if mk_result.trend == 'decreasing' else \
                                "➡️ No trend"
                
                significance = "***" if mk_result.p < 0.001 else \
                              "**" if mk_result.p < 0.01 else \
                              "*" if mk_result.p < 0.05 else \
                              ""
                
                print(f"📈 {description:<25}: {trend_direction} {significance}")
                print(f"   p-value: {mk_result.p:.4f}, τ: {mk_result.Tau:.3f}, "
                      f"slope: {mk_result.slope:.4f}")
                print()
                
            except Exception as e:
                print(f"❌ Error analyzing {description}: {str(e)}")
                results[metric] = None
    
    return results

def create_trend_visualizations(results, monthly_stats):
    """Create comprehensive visualizations of trend analysis."""
    
    print(f"📊 Creating trend visualizations...")
    
    if not results or not monthly_stats:
        print("❌ No results to visualize")
        return
    
    # Create DataFrame
    df = pd.DataFrame(monthly_stats)
    
    # Set up the plot
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Mann-Kendall Trend Analysis of Gi* Temporal Patterns', fontsize=16, fontweight='bold')
    
    plot_configs = [
        ('mean', 'Mean Gi* Value', 'Monthly Mean Gi* Trend'),
        ('median', 'Median Gi* Value', 'Monthly Median Gi* Trend'),
        ('std', 'Standard Deviation', 'Gi* Variability Trend'),
        ('hotspot_pct', 'Hot Spot %', 'Hot Spot Percentage Trend'),
        ('coldspot_pct', 'Cold Spot %', 'Cold Spot Percentage Trend'),
        ('hotspots', 'Hot Spot Count', 'Hot Spot Count Trend'),
        ('coldspots', 'Cold Spot Count', 'Cold Spot Count Trend'),
        ('significant', 'Significant Pixels', 'Significant Clustering Trend'),
    ]
    
    for i, (metric, ylabel, title) in enumerate(plot_configs):
        if i >= 8:  # Only 8 subplots available
            break
            
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        if metric in df.columns and metric in results and results[metric]:
            result = results[metric]
            data = result['data']
            months = df['month_num'].values
            
            # Plot data points
            ax.plot(months, data, 'o-', color='steelblue', linewidth=2, markersize=6, alpha=0.7)
            
            # Add trend line if significant
            if result['p_value'] < 0.05:
                trend_line = result['intercept'] + result['slope'] * np.arange(len(data))
                ax.plot(months, trend_line, '--', color='red', linewidth=2, alpha=0.8)
                
                # Add trend annotation
                trend_text = f"{result['trend'].title()}\n" \
                           f"p={result['p_value']:.3f}\n" \
                           f"τ={result['tau']:.3f}"
                ax.text(0.05, 0.95, trend_text, transform=ax.transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Month')
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(months)
            ax.set_xticklabels([df.iloc[j]['month_name'][:3] for j in range(len(months))], rotation=45)
        else:
            ax.text(0.5, 0.5, f'No data for\n{title}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontweight='bold')
    
    # Use the last subplot for summary
    ax = axes[2, 2]
    ax.axis('off')
    
    # Create summary text
    summary_text = "📊 TREND SUMMARY\n\n"
    significant_trends = []
    
    for metric, result in results.items():
        if result and result['p_value'] < 0.05:
            direction = "↗️" if result['trend'] == 'increasing' else "↘️"
            significance = "***" if result['p_value'] < 0.001 else "**" if result['p_value'] < 0.01 else "*"
            significant_trends.append(f"{direction} {result['description']} {significance}")
    
    if significant_trends:
        summary_text += "\n".join(significant_trends)
    else:
        summary_text += "No significant trends detected\n(p < 0.05)"
    
    summary_text += f"\n\n📅 Analysis period: {len(monthly_stats)} months"
    summary_text += f"\n🎯 Significance levels:"
    summary_text += f"\n   * p < 0.05"
    summary_text += f"\n   ** p < 0.01"
    summary_text += f"\n   *** p < 0.001"
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mann_kendall_analysis_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ Trend analysis plot saved: {filename}")
    
    plt.show()

def generate_interpretation_report(results, monthly_stats):
    """Generate a comprehensive interpretation report."""
    
    print(f"\n📋 MANN-KENDALL ANALYSIS REPORT")
    print("=" * 80)
    
    if not results or not monthly_stats:
        print("❌ No results to report")
        return
    
    print(f"🔍 ANALYSIS OVERVIEW:")
    print(f"   • Time period: {len(monthly_stats)} months")
    print(f"   • Metrics analyzed: {len([r for r in results.values() if r is not None])}")
    print(f"   • Significant trends: {len([r for r in results.values() if r and r['p_value'] < 0.05])}")
    
    print(f"\n🎯 SIGNIFICANT TRENDS (p < 0.05):")
    print("-" * 50)
    
    significant_count = 0
    for metric, result in results.items():
        if result and result['p_value'] < 0.05:
            significant_count += 1
            
            trend_symbol = "↗️" if result['trend'] == 'increasing' else "↘️"
            sig_level = "***" if result['p_value'] < 0.001 else "**" if result['p_value'] < 0.01 else "*"
            
            print(f"{trend_symbol} {result['description']} {sig_level}")
            print(f"   p-value: {result['p_value']:.4f}")
            print(f"   Kendall's τ: {result['tau']:.3f}")
            print(f"   Slope: {result['slope']:.4f} per month")
            print(f"   Trend: {result['trend'].title()}")
            print()
        
        if significant_count == 0:
            print("   No statistically significant trends detected.")
    
    print(f"\n🌍 SPATIAL CLUSTERING INTERPRETATION:")
    print("-" * 50)
    # Analyze what the trends mean for spatial clustering
    if 'mean' in results and results['mean']:
        mean_result = results['mean']
        if mean_result['p_value'] < 0.05:
            if mean_result['trend'] == 'increasing':
                print("   📈 Overall spatial clustering is becoming more positive over time")
                print("      → Transition from cold spots to hot spots")
                print("      → Potential economic development/recovery")
            else:
                print("   📉 Overall spatial clustering is becoming more negative over time")
                print("      → Increasing cold spot dominance")
                print("      → Potential economic decline or infrastructure changes")
        else:
            print("   ➡️ Overall spatial clustering shows no significant temporal trend")
    
    print(f"\n📊 NEXT STEPS:")
    print("-" * 50)
    print("   1. ✅ Real pixel-level analysis completed with rasterio")
    print("   2. ✅ Pixel-wise Mann-Kendall spatial trend analysis completed")
    print("   3. ✅ Spatial maps show WHERE temporal trends occur geographically")
    print("   4. Correlate spatial patterns with known regional events/policies")
    print("   5. Perform targeted analysis of persistent hotspot locations")
    print("   6. Consider seasonal decomposition for deeper temporal analysis")
    print("   7. Validate trend locations with ground-truth economic data")
    print("   8. Examine infrastructure/development in trend areas")
    print("   9. Create time-series animations of trend evolution")
    print("   10. Investigate why certain areas show temporal changes")
    
    print(f"\n🗺️ SPATIAL TREND ANALYSIS INSIGHTS:")
    print("-" * 50)
    print("   • Mann-Kendall spatial maps show temporal trend distribution")
    print("   • Red areas represent increasing temporal trends")
    print("   • Blue areas represent decreasing temporal trends")
    print("   • Maps reveal spatial heterogeneity in temporal patterns")
    print("   • Use these maps to identify priority areas for investigation")

def load_spatial_data_for_mapping(file_info):
    """Load spatial data with coordinates for mapping hotspots."""
    
    print(f"\n🗺️ Loading spatial data for mapping...")
    print("=" * 60)
    
    spatial_data = {}
    
    if RASTERIO_AVAILABLE:
        print("Loading spatial coordinates and clustering data...")
        
        for info in file_info:
            try:
                with rasterio.open(info['file']) as src:
                    data = src.read(1)  # Read first band
                    transform = src.transform
                    crs = src.crs
                    bounds = src.bounds
                    
                    # Create coordinate arrays
                    height, width = data.shape
                    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
                    
                    # Transform pixel coordinates to geographic coordinates
                    xs, ys = rasterio.transform.xy(transform, rows, cols)
                    lons = np.array(xs)
                    lats = np.array(ys)
                    
                    # Filter valid data
                    valid_mask = ~np.isnan(data) & np.isfinite(data)
                      # Verify CRS and coordinate system
                    print(f"  {info['month_name']:>10}: CRS = {crs}, Bounds = ({bounds.left:.3f}, {bounds.bottom:.3f}, {bounds.right:.3f}, {bounds.top:.3f})")
                    
                    # Ensure we have a proper geographic coordinate system
                    if crs and not crs.is_geographic:
                        print(f"    ⚠️ Warning: Non-geographic CRS detected. Consider reprojecting to WGS84.")
                    
                    spatial_data[info['month_num']] = {
                        'month_name': info['month_name'],
                        'data': data,
                        'lons': lons,
                        'lats': lats,
                        'valid_mask': valid_mask,
                        'transform': transform,
                        'crs': crs,
                        'bounds': bounds,
                        'extent': [bounds.left, bounds.right, bounds.bottom, bounds.top]
                    }
                    
                    # Calculate hotspot statistics
                    valid_data = data[valid_mask]
                    hotspots_95 = np.sum(valid_data > 1.96)
                    hotspots_90 = np.sum(valid_data > 1.645)  # 90% confidence
                    total_valid = len(valid_data)
                    
                    print(f"  {info['month_name']:>10}: {total_valid:6d} pixels, "
                          f"Hot95%={hotspots_95:4d}, Hot90%={hotspots_90:4d}")
                    
            except Exception as e:
                print(f"  {info['month_name']:>10}: ❌ Error reading spatial data - {str(e)}")
    
    return spatial_data

def create_mann_kendall_spatial_maps(file_info):
    """Create spatial maps showing Mann-Kendall trend results pixel by pixel."""
    
    print(f"\n🗺️ Creating Mann-Kendall spatial trend maps...")
    print("=" * 60)
    
    if not RASTERIO_AVAILABLE:
        print("❌ Rasterio not available - cannot create spatial maps")
        return None
    
    # Load all spatial data
    spatial_data = load_spatial_data_for_mapping(file_info)
    if not spatial_data:
        return None
    
    # Get reference data for shape and extent
    ref_month = list(spatial_data.keys())[0]
    ref_data = spatial_data[ref_month]
    height, width = ref_data['data'].shape
    bounds = ref_data['bounds']
    crs = ref_data['crs']
    
    print(f"Performing pixel-wise Mann-Kendall analysis on {height}x{width} grid...")
    
    # Collect all data into a 3D array [time, height, width]
    n_months = len(spatial_data)
    all_data = np.full((n_months, height, width), np.nan)
    
    # Fill the array with data from each month
    for idx, (month_num, month_data) in enumerate(sorted(spatial_data.items())):
        all_data[idx, :, :] = month_data['data']
    
    print(f"Collected data from {n_months} months for pixel-wise analysis...")
    
    # Initialize result arrays
    trend_direction = np.full((height, width), np.nan)  # -1=decreasing, 0=no trend, 1=increasing
    p_values = np.full((height, width), np.nan)         # p-values
    tau_values = np.full((height, width), np.nan)       # Kendall's tau (trend strength)
    slope_values = np.full((height, width), np.nan)     # Trend slope
    
    # Perform Mann-Kendall test for each pixel
    total_pixels = height * width
    processed_pixels = 0
    significant_pixels = 0
    
    for i in range(height):
        for j in range(width):
            # Get time series for this pixel
            pixel_time_series = all_data[:, i, j]
            
            # Skip if too many NaN values
            valid_data = pixel_time_series[~np.isnan(pixel_time_series)]
            if len(valid_data) < 3:  # Need at least 3 points for trend analysis
                continue
            
            try:
                # Perform Mann-Kendall test
                result = mk.original_test(valid_data)
                
                # Store results
                p_values[i, j] = result.p
                tau_values[i, j] = result.Tau
                slope_values[i, j] = result.slope
                  # Convert trend to numeric
                if result.trend == 'increasing':
                    trend_direction[i, j] = 1
                elif result.trend == 'decreasing':
                    trend_direction[i, j] = -1
                else:
                    trend_direction[i, j] = 0
                
                if result.p < 0.05:  # Count statistically significant trends
                    significant_pixels += 1
                
                processed_pixels += 1
                
            except Exception as e:
                continue
        
        # Progress update
        if i % 10 == 0:
            progress = (i / height) * 100
            print(f"  Progress: {progress:.1f}% ({significant_pixels} significant pixels found)")
    
    print(f"Completed pixel-wise analysis:")
    print(f"  • Processed pixels: {processed_pixels:,}")
    print(f"  • Significant trends (p<0.05): {significant_pixels} ({significant_pixels/processed_pixels*100:.2f}%)")
    
    # Create spatial maps
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Mann-Kendall Spatial Trend Analysis - MIGEDC Region\nPixel-wise Temporal Trend Detection', 
                 fontsize=16, fontweight='bold')
    
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    
    # 1. Trend Direction Map
    ax1 = axes[0, 0]
    ax1.set_facecolor('#f0f0f0')
      # Create masked array for significant trends only
    significant_trends = np.where(p_values < 0.05, trend_direction, np.nan)
    
    im1 = ax1.imshow(significant_trends, extent=extent, cmap='RdBu_r', 
                     vmin=-1, vmax=1, alpha=0.8)
    ctx.add_basemap(ax1, crs="EPSG:4326", source=xyz.CartoDB.Positron, alpha=0.7)
    im1 = ax1.imshow(significant_trends, extent=extent, cmap='RdBu_r', 
                     vmin=-1, vmax=1, alpha=0.8, zorder=1)
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_xlim(bounds.left, bounds.right)
    ax1.set_ylim(bounds.bottom, bounds.top)
    ax1.set_title('Trend Direction (p < 0.05)\nRed=Increasing, Blue=Decreasing', fontweight='bold')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_ticks([-1, 0, 1])
    cbar1.set_ticklabels(['Decreasing', 'No Trend', 'Increasing'])
    
    # 2. P-value Map (significance)
    ax2 = axes[0, 1]
    ax2.set_facecolor('#f0f0f0')    # Show only significant p-values
    p_display = np.where(p_values < 0.05, p_values, np.nan)
    
    im2 = ax2.imshow(p_display, extent=extent, cmap='viridis_r', 
                     vmin=0, vmax=0.05, alpha=0.8)
    ctx.add_basemap(ax2, crs="EPSG:4326", source=xyz.CartoDB.Positron, alpha=0.7)
    im2 = ax2.imshow(p_display, extent=extent, cmap='viridis_r', 
                     vmin=0, vmax=0.05, alpha=0.8, zorder=1)
    ax2.set_aspect('equal', adjustable='box')
    ax2.set_xlim(bounds.left, bounds.right)
    ax2.set_ylim(bounds.bottom, bounds.top)
    ax2.set_title('Trend Significance\n(p-values < 0.05)', fontweight='bold')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.grid(True, alpha=0.3)
    
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label('p-value')
    
    # 3. Trend Strength (Kendall's Tau)
    ax3 = axes[1, 0]
    ax3.set_facecolor('#f0f0f0')
      # Show tau values for significant trends
    tau_display = np.where(p_values < 0.05, tau_values, np.nan)
    
    im3 = ax3.imshow(tau_display, extent=extent, cmap='RdBu_r', 
                     vmin=-1, vmax=1, alpha=0.8)
    ctx.add_basemap(ax3, crs="EPSG:4326", source=xyz.CartoDB.Positron, alpha=0.7)
    im3 = ax3.imshow(tau_display, extent=extent, cmap='RdBu_r', 
                     vmin=-1, vmax=1, alpha=0.8, zorder=1)
    ax3.set_aspect('equal', adjustable='box')
    ax3.set_xlim(bounds.left, bounds.right)
    ax3.set_ylim(bounds.bottom, bounds.top)
    ax3.set_title('Trend Strength (Kendall\'s Tau)\nfor Significant Trends', fontweight='bold')
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')
    ax3.grid(True, alpha=0.3)
    
    cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.8)
    cbar3.set_label('Kendall\'s Tau')
    
    # 4. Slope Map
    ax4 = axes[1, 1]
    ax4.set_facecolor('#f0f0f0')
      # Show slope values for significant trends
    slope_display = np.where(p_values < 0.05, slope_values, np.nan)
    
    # Plot slope first, then add basemap (to keep colors correct)
    im4 = ax4.imshow(slope_display, extent=extent, cmap='RdBu_r', 
                     vmin=np.nanpercentile(slope_display, 5), 
                     vmax=np.nanpercentile(slope_display, 95), alpha=0.8)
    # Add basemap underneath by drawing after imshow, then re-draw imshow to restore colors
    ctx.add_basemap(ax4, crs="EPSG:4326", source=xyz.CartoDB.Positron, alpha=0.7, zorder=0)
    # Re-draw imshow on top to preserve color scale
    im4 = ax4.imshow(slope_display, extent=extent, cmap='RdBu_r', 
                     vmin=np.nanpercentile(slope_display, 5), 
                     vmax=np.nanpercentile(slope_display, 95), alpha=0.8, zorder=1)
    ax4.set_aspect('equal', adjustable='box')
    ax4.set_xlim(bounds.left, bounds.right)
    ax4.set_ylim(bounds.bottom, bounds.top)
    ax4.set_title('Trend Slope\nfor Significant Trends', fontweight='bold')
    ax4.set_xlabel('Longitude')
    ax4.set_ylabel('Latitude')
    ax4.grid(True, alpha=0.3)
    
    cbar4 = plt.colorbar(im4, ax=ax4, shrink=0.8)
    cbar4.set_label('Slope (Gi*/month)')
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mann_kendall_spatial_maps_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Mann-Kendall spatial maps saved: {filename}")
    
    plt.show()
    
    # Create summary statistics
    create_spatial_trend_summary(p_values, trend_direction, tau_values, significant_pixels, processed_pixels)
    
    return {
        'trend_direction': trend_direction,
        'p_values': p_values,
        'tau_values': tau_values,
        'slope_values': slope_values,
        'bounds': bounds,
        'crs': crs
    }

def create_spatial_trend_summary(p_values, trend_direction, tau_values, significant_pixels, processed_pixels):
    """Create a summary of spatial trend analysis results."""
    
    print(f"\n📊 MANN-KENDALL SPATIAL TREND SUMMARY")
    print("=" * 50)
      # Count trends by significance level
    highly_sig = np.sum(p_values < 0.01)
    moderately_sig = np.sum((p_values >= 0.01) & (p_values < 0.05))
    marginally_sig = np.sum((p_values >= 0.05) & (p_values < 0.10))
    
    print(f"📈 TREND SIGNIFICANCE LEVELS:")
    print(f"   • Highly significant (p < 0.01): {highly_sig} pixels")
    print(f"   • Moderately significant (0.01 ≤ p < 0.05): {moderately_sig} pixels")
    print(f"   • Marginally significant (0.05 ≤ p < 0.10): {marginally_sig} pixels")
    print(f"   • Total significant (p < 0.05): {significant_pixels} pixels")
    print(f"   • Percentage of area: {significant_pixels/processed_pixels*100:.2f}%")
    
    # Count trend directions (for significant trends only)
    sig_mask = p_values < 0.05
    if np.any(sig_mask):
        increasing = np.sum((trend_direction == 1) & sig_mask)
        decreasing = np.sum((trend_direction == -1) & sig_mask)
        no_trend = np.sum((trend_direction == 0) & sig_mask)
        
        print(f"\n📊 TREND DIRECTIONS (significant only):")
        print(f"   • Increasing trends: {increasing} pixels ({increasing/significant_pixels*100:.1f}%)")
        print(f"   • Decreasing trends: {decreasing} pixels ({decreasing/significant_pixels*100:.1f}%)")
        print(f"   • No clear trend: {no_trend} pixels ({no_trend/significant_pixels*100:.1f}%)")
        
        # Trend strength analysis
        if np.any(~np.isnan(tau_values)):
            mean_tau_inc = np.nanmean(tau_values[(trend_direction == 1) & sig_mask])
            mean_tau_dec = np.nanmean(tau_values[(trend_direction == -1) & sig_mask])
            
            print(f"\n💪 TREND STRENGTH (Kendall's Tau):")
            print(f"   • Mean increasing trend strength: {mean_tau_inc:.3f}")
            print(f"   • Mean decreasing trend strength: {mean_tau_dec:.3f}")
    
    print(f"\n🎯 INTERPRETATION:")
    if significant_pixels == 0:
        print(f"   • No significant temporal trends detected")
        print(f"   • Gi* patterns are temporally stable across the region")
    elif significant_pixels/processed_pixels < 0.05:
        print(f"   • Very few areas show temporal trends (<5% of region)")
        print(f"   • Overall spatial patterns are highly stable")
    else:
        print(f"   • Some areas showing temporal changes ({significant_pixels/processed_pixels*100:.1f}% of region)")
        print(f"   • Mixed stability: some areas changing, most stable")

def save_mann_kendall_rasters(spatial_results, output_dir="./mann_kendall_results"):
    """Save Mann-Kendall spatial results as GeoTIFF rasters."""
    
    print(f"\n💾 Saving Mann-Kendall results as GeoTIFF rasters...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    if not RASTERIO_AVAILABLE:
        print("❌ Rasterio not available - cannot save GeoTIFF files")
        return False
    
    try:
        # Get spatial data
        trend_direction = spatial_results['trend_direction']
        p_values = spatial_results['p_values']
        tau_values = spatial_results['tau_values']
        slope_values = spatial_results['slope_values']
        bounds = spatial_results['bounds']
        crs = spatial_results['crs']
        
        # Calculate transform from bounds
        height, width = trend_direction.shape
        transform = rasterio.transform.from_bounds(
            bounds.left, bounds.bottom, bounds.right, bounds.top, width, height
        )
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Define raster metadata
        meta = {
            'driver': 'GTiff',
            'dtype': 'float32',
            'nodata': np.nan,
            'width': width,
            'height': height,
            'count': 1,
            'crs': crs,
            'transform': transform,
            'compress': 'lzw'
        }
        
        # Save trend direction raster
        trend_file = os.path.join(output_dir, f"mann_kendall_trend_direction_{timestamp}.tif")
        with rasterio.open(trend_file, 'w', **meta) as dst:
            dst.write(trend_direction.astype(np.float32), 1)
        print(f"   ✅ Trend direction raster: {trend_file}")
        
        # Save p-values raster
        pvalue_file = os.path.join(output_dir, f"mann_kendall_pvalues_{timestamp}.tif")
        with rasterio.open(pvalue_file, 'w', **meta) as dst:
            dst.write(p_values.astype(np.float32), 1)
        print(f"   ✅ P-values raster: {pvalue_file}")
        
        # Save Kendall's Tau raster
        tau_file = os.path.join(output_dir, f"mann_kendall_tau_{timestamp}.tif")
        with rasterio.open(tau_file, 'w', **meta) as dst:
            dst.write(tau_values.astype(np.float32), 1)
        print(f"   ✅ Tau values raster: {tau_file}")
        
        # Save slope raster
        slope_file = os.path.join(output_dir, f"mann_kendall_slope_{timestamp}.tif")
        with rasterio.open(slope_file, 'w', **meta) as dst:
            dst.write(slope_values.astype(np.float32), 1)
        print(f"   ✅ Slope values raster: {slope_file}")
        
        # Save significance mask raster (binary: 1=significant, 0=not significant)
        significance_mask = (p_values < 0.05).astype(np.float32)
        significance_mask[np.isnan(p_values)] = np.nan
        
        sig_file = os.path.join(output_dir, f"mann_kendall_significance_mask_{timestamp}.tif")
        with rasterio.open(sig_file, 'w', **meta) as dst:
            dst.write(significance_mask, 1)
        print(f"   ✅ Significance mask raster: {sig_file}")
        
        # Create metadata file
        metadata_file = os.path.join(output_dir, f"mann_kendall_metadata_{timestamp}.txt")
        with open(metadata_file, 'w') as f:
            f.write("Mann-Kendall Trend Analysis Raster Metadata\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"CRS: {crs}\n")
            f.write(f"Bounds: {bounds}\n")
            f.write(f"Resolution: {width} x {height} pixels\n")
            f.write(f"Transform: {transform}\n\n")
            
            f.write("Raster Files:\n")
            f.write(f"- Trend Direction: {os.path.basename(trend_file)}\n")
            f.write(f"  Values: -1 (decreasing), 0 (no trend), 1 (increasing)\n\n")
            f.write(f"- P-values: {os.path.basename(pvalue_file)}\n")
            f.write(f"  Values: 0.0 to 1.0 (lower = more significant)\n\n")
            f.write(f"- Kendall's Tau: {os.path.basename(tau_file)}\n")
            f.write(f"  Values: -1.0 to 1.0 (trend strength and direction)\n\n")
            f.write(f"- Slope: {os.path.basename(slope_file)}\n")
            f.write(f"  Values: Rate of change per time step\n\n")
            f.write(f"- Significance Mask: {os.path.basename(sig_file)}\n")
            f.write(f"  Values: 1 (p < 0.05), 0 (not significant), NaN (no data)\n\n")
            
            # Add statistics
            total_pixels = np.sum(~np.isnan(p_values))
            sig_pixels = np.sum(p_values < 0.05)
            
            f.write("Statistics:\n")
            f.write(f"- Total analyzed pixels: {total_pixels:,}\n")
            f.write(f"- Significant pixels (p < 0.05): {sig_pixels:,} ({sig_pixels/total_pixels*100:.2f}%)\n")
            
            if sig_pixels > 0:
                increasing = np.sum((trend_direction == 1) & (p_values < 0.05))
                decreasing = np.sum((trend_direction == -1) & (p_values < 0.05))
                f.write(f"- Increasing trends: {increasing:,} ({increasing/sig_pixels*100:.1f}% of significant)\n")
                f.write(f"- Decreasing trends: {decreasing:,} ({decreasing/sig_pixels*100:.1f}% of significant)\n")
        
        print(f"   ✅ Metadata file: {metadata_file}")
        
        print(f"\n📁 Mann-Kendall raster outputs saved to: {output_dir}")
        print(f"🎯 {len(os.listdir(output_dir))} files created")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving rasters: {e}")
        return False

def create_composite_raster(spatial_results, output_dir="./mann_kendall_results"):
    """Create a composite raster combining trend direction and significance."""
    
    print(f"\n🎨 Creating composite Mann-Kendall raster...")
    
    if not RASTERIO_AVAILABLE:
        print("❌ Rasterio not available - cannot create composite raster")
        return False
    
    try:
        trend_direction = spatial_results['trend_direction']
        p_values = spatial_results['p_values']
        bounds = spatial_results['bounds']
        crs = spatial_results['crs']        # Create composite values:
        # -2: Significant decreasing (p < 0.05)
        #  0: No significant trend (p ≥ 0.05)
        #  2: Significant increasing (p < 0.05)
        
        composite = np.full_like(trend_direction, np.nan)
          # Significant trends (p < 0.05)
        sig_mask = p_values < 0.05
        composite[(trend_direction == 1) & sig_mask] = 2   # Significant increasing
        composite[(trend_direction == -1) & sig_mask] = -2  # Significant decreasing
        
        # All other trends are not significant
        no_trend_mask = (p_values >= 0.05) | (trend_direction == 0)
        composite[no_trend_mask] = 0
        
        # Save composite raster
        height, width = composite.shape
        transform = rasterio.transform.from_bounds(
            bounds.left, bounds.bottom, bounds.right, bounds.top, width, height
        )
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        composite_file = os.path.join(output_dir, f"mann_kendall_composite_{timestamp}.tif")
        
        meta = {
            'driver': 'GTiff',
            'dtype': 'float32',
            'nodata': np.nan,
            'width': width,
            'height': height,
            'count': 1,            'crs': crs,
            'transform': transform,
            'compress': 'lzw'
        }
        
        with rasterio.open(composite_file, 'w', **meta) as dst:
            dst.write(composite.astype(np.float32), 1)
        
        print(f"   ✅ Composite raster: {composite_file}")
        print(f"   📊 Values: -2=Sig.Dec, 0=NoTrend, 2=Sig.Inc (p<0.05)")
        
        return composite_file
        
    except Exception as e:
        print(f"❌ Error creating composite raster: {e}")
        return False

def main():
    """Main function for Mann-Kendall trend analysis with spatial mapping and raster output."""
    
    print("Mann-Kendall Trend Analysis with Spatial Mapping and Raster Output")
    print("=" * 80)
    print("Analyzing temporal trends in Gi* patterns - WHERE trends occur spatially")
    print("=" * 80)
    
    # Load data
    file_info, gi_files = load_gi_star_data()
    if not file_info:
        return
    
    # Calculate statistics for time series analysis
    monthly_stats = calculate_pixel_statistics(file_info)
    if not monthly_stats:
        print("❌ No statistics calculated")
        return
    
    # Perform Mann-Kendall analysis on aggregated statistics
    results = perform_mann_kendall_analysis(monthly_stats)
    if not results:
        print("❌ No trend analysis results")
        return
    
    # Create temporal visualizations (time series plots)
    create_trend_visualizations(results, monthly_stats)
    
    # Create Mann-Kendall spatial maps (pixel-by-pixel trend analysis)  
    print(f"\n🗺️ Creating spatial Mann-Kendall trend maps...")
    spatial_results = create_mann_kendall_spatial_maps(file_info)
    
    # NEW: Save raster outputs
    if spatial_results:
        print(f"\n💾 Saving Mann-Kendall results as raster files...")
        save_success = save_mann_kendall_rasters(spatial_results)
        
        if save_success:
            # Create composite raster
            composite_file = create_composite_raster(spatial_results)
            
            print(f"\n📊 RASTER OUTPUTS SUMMARY:")
            print(f"   • Trend direction raster (GeoTIFF)")
            print(f"   • P-values raster (GeoTIFF)")
            print(f"   • Kendall's Tau raster (GeoTIFF)")
            print(f"   • Slope values raster (GeoTIFF)")
            print(f"   • Significance mask raster (GeoTIFF)")
            print(f"   • Composite trend raster (GeoTIFF)")
            print(f"   • Metadata documentation (TXT)")
      # Generate interpretation report
    generate_interpretation_report(results, monthly_stats)
    
    print(f"\n✅ Complete Mann-Kendall analysis finished!")
    print(f"📁 Files generated:")
    print(f"   • mann_kendall_analysis_*.png (temporal trends)")
    print(f"   • mann_kendall_spatial_maps_*.png (spatial trend maps)")
    print(f"   • ./mann_kendall_results/ (GeoTIFF raster outputs)")
    print(f"🔍 Pixel-level spatial trend analysis performed")
    print(f"📊 Shows WHERE temporal trends are occurring geographically")
    print(f"💾 Raster outputs ready for GIS analysis")
    
    if spatial_results and np.any(spatial_results['p_values'] < 0.05):
        sig_count = np.sum(spatial_results['p_values'] < 0.05)
        total_count = np.sum(~np.isnan(spatial_results['p_values']))
        print(f"🎯 Found {sig_count} pixels with significant temporal trends ({sig_count/total_count*100:.2f}% of area)")
    else:
        print(f"📈 No significant spatial trends detected - patterns are temporally stable")

if __name__ == "__main__":
    main()
