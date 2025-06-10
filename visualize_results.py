#!/usr/bin/env python3
"""
Visualization script for Getis-Ord Gi* analysis results
"""

import ee
import yaml
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import glob
import os
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
from datetime import datetime
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
        raise

def create_gi_star_colormap():
    """Create colormap for Gi* visualization."""
    # Colors: dark blue (cold spots) -> white (neutral) -> dark red (hot spots)
    colors = ['#08306b', '#2171b5', '#6baed6', '#c6dbef', '#f7f7f7', 
              '#fdd49e', '#fc8d59', '#e34a33', '#b30000']
    return ListedColormap(colors)

def create_significance_colormap():
    """Create colormap for significance classification."""
    # Colors for significance levels: -2, -1, 0, 1, 2
    colors = ['#08306b', '#6baed6', '#f7f7f7', '#fc8d59', '#b30000']
    return ListedColormap(colors)

def visualize_results(config):
    """
    Create visualization maps for the Getis-Ord Gi* analysis results.
    This function demonstrates how to visualize the exported results.
    """
    
    # Initialize Earth Engine
    initialize_ee(config['ee']['project'])
    
    # Define ROI
    roi = ee.Geometry.Rectangle(config['ee']['roi'])
    
    # For demonstration, we'll create sample visualization parameters
    # In practice, you would load your exported images
    
    print("Visualization Parameters:")
    print("=" * 40)
    
    # Gi* visualization parameters
    gi_star_vis = {
        'min': -3,
        'max': 3,
        'palette': ['08306b', '2171b5', '6baed6', 'c6dbef', 'f7f7f7', 
                   'fdd49e', 'fc8d59', 'e34a33', 'b30000']
    }
    
    # Z-score visualization parameters
    z_score_vis = {
        'min': -4,
        'max': 4,
        'palette': ['08306b', '2171b5', '6baed6', 'c6dbef', 'f7f7f7', 
                   'fdd49e', 'fc8d59', 'e34a33', 'b30000']
    }
    
    # Significance classification visualization
    significance_vis = {
        'min': -2,
        'max': 2,
        'palette': ['08306b', '6baed6', 'f7f7f7', 'fc8d59', 'b30000']
    }
    
    # NTL change visualization
    ntl_change_vis = {
        'min': -5,
        'max': 5,
        'palette': ['440154', '31688e', '35b779', 'fde725']
    }
    
    print("\nGi* Statistics Visualization:")
    print(f"  Min: {gi_star_vis['min']}, Max: {gi_star_vis['max']}")
    print(f"  Palette: {gi_star_vis['palette']}")
    
    print("\nZ-Score Visualization:")
    print(f"  Min: {z_score_vis['min']}, Max: {z_score_vis['max']}")
    print(f"  Palette: {z_score_vis['palette']}")
    
    print("\nSignificance Classification Visualization:")
    print(f"  Min: {significance_vis['min']}, Max: {significance_vis['max']}")
    print(f"  Palette: {significance_vis['palette']}")
    print("  Classes: -2 (Very Cold), -1 (Cold), 0 (Not Significant), 1 (Hot), 2 (Very Hot)")
    
    print("\nNighttime Lights Change Visualization:")
    print(f"  Min: {ntl_change_vis['min']}, Max: {ntl_change_vis['max']}")
    print(f"  Palette: {ntl_change_vis['palette']}")
    
    # Create legend information
    create_legend_info()

def create_legend_info():
    """Create legend information for the visualizations."""
    
    print("\n" + "="*60)
    print("LEGEND INFORMATION")
    print("="*60)
    
    print("\n1. Getis-Ord Gi* Statistics:")
    print("   • Dark Blue: Strong cold spots (Gi* < -2)")
    print("   • Light Blue: Moderate cold spots (-2 ≤ Gi* < -1)")
    print("   • White/Gray: No clustering (-1 ≤ Gi* ≤ 1)")
    print("   • Light Red: Moderate hot spots (1 < Gi* ≤ 2)")
    print("   • Dark Red: Strong hot spots (Gi* > 2)")
    
    print("\n2. Z-Score Significance:")
    print("   • Dark Blue: Highly significant cold spots (z < -2.58)")
    print("   • Light Blue: Significant cold spots (-2.58 ≤ z < -1.96)")
    print("   • White/Gray: Not significant (-1.96 ≤ z ≤ 1.96)")
    print("   • Light Red: Significant hot spots (1.96 < z ≤ 2.58)")
    print("   • Dark Red: Highly significant hot spots (z > 2.58)")
    
    print("\n3. Significance Classification:")
    print("   • -2: Very significant cold spots (p < 0.01)")
    print("   • -1: Significant cold spots (p < 0.05)")
    print("   •  0: Not statistically significant")
    print("   •  1: Significant hot spots (p < 0.05)")
    print("   •  2: Very significant hot spots (p < 0.01)")
    
    print("\n4. Interpretation Guide:")
    print("   Hot Spots (Positive Gi*, Red colors):")
    print("   • Areas with high nighttime lights surrounded by high values")
    print("   • Indicate economic activity clusters or urban centers")
    print("   • May represent commercial districts or industrial areas")
    
    print("\n   Cold Spots (Negative Gi*, Blue colors):")
    print("   • Areas with low nighttime lights surrounded by low values")
    print("   • Indicate rural or underdeveloped regions")
    print("   • May represent agricultural or conservation areas")
    
    print("\n   Statistical Significance:")
    print("   • p < 0.05: 95% confidence that clustering is not random")
    print("   • p < 0.01: 99% confidence that clustering is not random")
    print("   • Higher confidence = more reliable spatial patterns")

def load_gi_star_data(data_dir="./data"):
    """Load all Gi* GeoTIFF files and return data dictionary."""
    
    gi_star_files = glob.glob(os.path.join(data_dir, "gi_star_*.tif"))
    gi_star_files.sort()  # Sort by filename for consistent ordering
    
    if not gi_star_files:
        print(f"❌ No Gi* files found in {data_dir}")
        return None
    
    gi_star_data = {}
    
    print(f"📂 Loading Gi* data from {len(gi_star_files)} files...")
    
    for file_path in gi_star_files:
        try:
            # Extract month info from filename
            filename = os.path.basename(file_path)
            parts = filename.replace('.tif', '').split('_')
            
            if len(parts) >= 4 and parts[0] == 'gi' and parts[1] == 'star':
                month_num = int(parts[2])  # Extract month number
                month_name = parts[3]     # Extract month name
                year = parts[4] if len(parts) > 4 else "2024"
                
                # Load raster data
                with rasterio.open(file_path) as src:
                    data = src.read(1)  # Read first band
                    transform = src.transform
                    crs = src.crs
                    bounds = src.bounds
                    
                    # Replace nodata values with NaN
                    if src.nodata is not None:
                        data = np.where(data == src.nodata, np.nan, data)
                    
                    gi_star_data[month_num] = {
                        'data': data,
                        'month_name': month_name,
                        'year': year,
                        'filename': filename,
                        'valid_pixels': np.sum(~np.isnan(data)),
                        'transform': transform,
                        'crs': crs,
                        'bounds': bounds
                    }
                    
                    print(f"   ✓ {month_name}: {data.shape} pixels, {np.sum(~np.isnan(data))} valid")
                    
        except Exception as e:
            print(f"   ❌ Error loading {filename}: {e}")
    
    if gi_star_data:
        print(f"✅ Successfully loaded {len(gi_star_data)} months of Gi* data")
        return gi_star_data
    else:
        print("❌ No valid Gi* data loaded")
        return None

def analyze_gi_star_statistics(gi_star_data):
    """Analyze statistics across all months."""
    
    print(f"\n📊 COMPREHENSIVE Gi* STATISTICS ANALYSIS")
    print("=" * 60)
    
    monthly_stats = {}
    all_values = []
    
    # Calculate statistics for each month
    for month_num in sorted(gi_star_data.keys()):
        data_info = gi_star_data[month_num]
        data = data_info['data']
        valid_data = data[~np.isnan(data)]
        
        if len(valid_data) > 0:
            stats = {
                'min': np.min(valid_data),
                'max': np.max(valid_data),
                'mean': np.mean(valid_data),
                'std': np.std(valid_data),
                'median': np.median(valid_data),
                'q25': np.percentile(valid_data, 25),
                'q75': np.percentile(valid_data, 75),
                'count': len(valid_data),
                'hot_spots_99': np.sum(valid_data > 2.58),   # 99% confidence
                'hot_spots_95': np.sum(valid_data > 1.96),   # 95% confidence
                'cold_spots_99': np.sum(valid_data < -2.58), # 99% confidence
                'cold_spots_95': np.sum(valid_data < -1.96), # 95% confidence
                'not_significant': np.sum((valid_data >= -1.96) & (valid_data <= 1.96))
            }
            
            monthly_stats[month_num] = stats
            all_values.extend(valid_data)
            
            print(f"\n{data_info['month_name']} ({month_num:02d}):")
            print(f"   Range: {stats['min']:.3f} to {stats['max']:.3f}")
            print(f"   Mean: {stats['mean']:.3f}, Median: {stats['median']:.3f}")
            print(f"   Std Dev: {stats['std']:.3f}")
            print(f"   Hot spots (99%): {stats['hot_spots_99']} ({stats['hot_spots_99']/stats['count']*100:.2f}%)")
            print(f"   Hot spots (95%): {stats['hot_spots_95']} ({stats['hot_spots_95']/stats['count']*100:.2f}%)")
            print(f"   Cold spots (99%): {stats['cold_spots_99']} ({stats['cold_spots_99']/stats['count']*100:.2f}%)")
            print(f"   Cold spots (95%): {stats['cold_spots_95']} ({stats['cold_spots_95']/stats['count']*100:.2f}%)")
    
    # Overall statistics
    if all_values:
        print(f"\n🌟 OVERALL STATISTICS (All Months Combined):")
        print(f"   Total valid pixels: {len(all_values):,}")
        print(f"   Overall range: {np.min(all_values):.3f} to {np.max(all_values):.3f}")
        print(f"   Overall mean: {np.mean(all_values):.3f}")
        print(f"   Overall std dev: {np.std(all_values):.3f}")
        print(f"   Overall hot spots (99%): {np.sum(np.array(all_values) > 2.58)} ({np.sum(np.array(all_values) > 2.58)/len(all_values)*100:.2f}%)")
        print(f"   Overall cold spots (99%): {np.sum(np.array(all_values) < -2.58)} ({np.sum(np.array(all_values) < -2.58)/len(all_values)*100:.2f}%)")
    
    return monthly_stats, all_values

def create_monthly_comparison_plot(gi_star_data, monthly_stats):
    """Create comprehensive monthly comparison plots."""
    
    months = sorted(gi_star_data.keys())
    month_names = [gi_star_data[m]['month_name'] for m in months]
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle('Monthly Gi* Analysis - MIGEDC Region', fontsize=16, fontweight='bold')
    
    # 1. Mean Gi* values by month
    means = [monthly_stats[m]['mean'] for m in months]
    axes[0,0].plot(months, means, 'o-', linewidth=2, markersize=8, color='darkblue')
    axes[0,0].set_title('Mean Gi* Value by Month')
    axes[0,0].set_xlabel('Month')
    axes[0,0].set_ylabel('Mean Gi*')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].set_xticks(months)
    axes[0,0].set_xticklabels([name[:3] for name in month_names], rotation=45)
    axes[0,0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 2. Hot/Cold spots by month
    hot_spots_99 = [monthly_stats[m]['hot_spots_99'] for m in months]
    hot_spots_95 = [monthly_stats[m]['hot_spots_95'] for m in months]
    cold_spots_99 = [monthly_stats[m]['cold_spots_99'] for m in months]
    cold_spots_95 = [monthly_stats[m]['cold_spots_95'] for m in months]
    
    x_pos = np.arange(len(months))
    width = 0.2
    
    axes[0,1].bar(x_pos - width*1.5, hot_spots_99, width, label='Hot (99%)', color='darkred', alpha=0.8)
    axes[0,1].bar(x_pos - width*0.5, hot_spots_95, width, label='Hot (95%)', color='red', alpha=0.6)
    axes[0,1].bar(x_pos + width*0.5, cold_spots_95, width, label='Cold (95%)', color='blue', alpha=0.6)
    axes[0,1].bar(x_pos + width*1.5, cold_spots_99, width, label='Cold (99%)', color='darkblue', alpha=0.8)
    
    axes[0,1].set_title('Significant Clustering by Month')
    axes[0,1].set_xlabel('Month')
    axes[0,1].set_ylabel('Number of Pixels')
    axes[0,1].set_xticks(x_pos)
    axes[0,1].set_xticklabels([name[:3] for name in month_names], rotation=45)
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Range (min/max) by month
    mins = [monthly_stats[m]['min'] for m in months]
    maxs = [monthly_stats[m]['max'] for m in months]
    
    axes[1,0].fill_between(months, mins, maxs, alpha=0.3, color='gray', label='Range')
    axes[1,0].plot(months, mins, 'o-', color='blue', label='Minimum')
    axes[1,0].plot(months, maxs, 'o-', color='red', label='Maximum')
    axes[1,0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1,0].set_title('Gi* Range by Month')
    axes[1,0].set_xlabel('Month')
    axes[1,0].set_ylabel('Gi* Value')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].set_xticks(months)
    axes[1,0].set_xticklabels([name[:3] for name in month_names], rotation=45)
    
    # 4. Standard deviation by month
    stds = [monthly_stats[m]['std'] for m in months]
    axes[1,1].plot(months, stds, 'o-', linewidth=2, markersize=8, color='green')
    axes[1,1].set_title('Gi* Variability by Month')
    axes[1,1].set_xlabel('Month')
    axes[1,1].set_ylabel('Standard Deviation')
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].set_xticks(months)
    axes[1,1].set_xticklabels([name[:3] for name in month_names], rotation=45)
    
    # 5. Percentage of significant pixels
    sig_pcts = [(monthly_stats[m]['hot_spots_95'] + monthly_stats[m]['cold_spots_95']) / monthly_stats[m]['count'] * 100 for m in months]
    axes[2,0].plot(months, sig_pcts, 'o-', linewidth=2, markersize=8, color='purple')
    axes[2,0].set_title('Significant Clustering Percentage')
    axes[2,0].set_xlabel('Month')
    axes[2,0].set_ylabel('% of Pixels Significant (95%)')
    axes[2,0].grid(True, alpha=0.3)
    axes[2,0].set_xticks(months)
    axes[2,0].set_xticklabels([name[:3] for name in month_names], rotation=45)
    
    # 6. Distribution plot for all months combined
    all_values = []
    for month_num in months:
        data = gi_star_data[month_num]['data']
        valid_data = data[~np.isnan(data)]
        all_values.extend(valid_data)
    
    axes[2,1].hist(all_values, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    axes[2,1].axvline(x=1.96, color='red', linestyle='--', alpha=0.7, label='95% confidence')
    axes[2,1].axvline(x=-1.96, color='red', linestyle='--', alpha=0.7)
    axes[2,1].axvline(x=2.58, color='darkred', linestyle='--', alpha=0.7, label='99% confidence')
    axes[2,1].axvline(x=-2.58, color='darkred', linestyle='--', alpha=0.7)
    axes[2,1].axvline(x=0, color='black', linestyle='-', alpha=0.5)
    axes[2,1].set_title('Overall Gi* Distribution')
    axes[2,1].set_xlabel('Gi* Value')
    axes[2,1].set_ylabel('Frequency')
    axes[2,1].legend()
    axes[2,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"gi_star_monthly_analysis_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ Monthly analysis plot saved: {filename}")
    
    plt.show()

def create_spatial_maps(gi_star_data, months_to_plot=[1, 4, 7, 10]):
    """Create spatial maps for selected months."""
    
    print(f"\n🗺️ Creating spatial maps for months: {months_to_plot}")
    
    # Filter available months
    available_months = [m for m in months_to_plot if m in gi_star_data.keys()]
    
    if not available_months:
        print("❌ No data available for requested months")
        return
    
    # Create figure
    n_months = len(available_months)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Spatial Distribution of Gi* Values - Selected Months', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    # Custom colormap for Gi*
    colors = ['#08306b', '#2171b5', '#6baed6', '#c6dbef', '#f7f7f7', 
              '#fdd49e', '#fc8d59', '#e34a33', '#b30000']
    cmap = ListedColormap(colors)
    
    for i, month_num in enumerate(available_months[:4]):  # Only plot first 4
        if i >= 4:
            break
            
        data_info = gi_star_data[month_num]
        data = data_info['data']
        
        # Plot the data
        im = axes[i].imshow(data, cmap=cmap, vmin=-5, vmax=5, aspect='auto')
        axes[i].set_title(f"{data_info['month_name']} ({month_num:02d})")
        axes[i].set_xlabel('Longitude')
        axes[i].set_ylabel('Latitude')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[i], shrink=0.8)
        cbar.set_label('Gi* Value')
    
    # Hide any unused subplots
    for i in range(len(available_months), 4):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"gi_star_spatial_maps_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ Spatial maps saved: {filename}")
    
    plt.show()

def create_significance_summary(monthly_stats):
    """Create a summary table of significance levels."""
    
    print(f"\n📋 SIGNIFICANCE SUMMARY TABLE")
    print("=" * 80)
    
    print(f"{'Month':<10} {'Mean':<8} {'Hot95%':<8} {'Hot99%':<8} {'Cold95%':<8} {'Cold99%':<8} {'SigTotal%':<10}")
    print("-" * 80)
    
    for month_num in sorted(monthly_stats.keys()):
        stats = monthly_stats[month_num]
        month_name = month_num  # You might want to use month names here
        
        hot95_pct = (stats['hot_spots_95'] / stats['count']) * 100
        hot99_pct = (stats['hot_spots_99'] / stats['count']) * 100
        cold95_pct = (stats['cold_spots_95'] / stats['count']) * 100
        cold99_pct = (stats['cold_spots_99'] / stats['count']) * 100
        sig_total_pct = hot95_pct + cold95_pct
        
        print(f"{month_name:<10} {stats['mean']:<8.3f} {hot95_pct:<8.2f} {hot99_pct:<8.2f} "
              f"{cold95_pct:<8.2f} {cold99_pct:<8.2f} {sig_total_pct:<10.2f}")
    
    print("-" * 80)
    print("Legend: Hot95% = Hot spots (95% confidence), Cold95% = Cold spots (95% confidence)")
    print("        SigTotal% = Total significant pixels (95% confidence)")

def interpret_gi_star_results(monthly_stats, all_values):
    """Provide comprehensive interpretation of Gi* results."""
    
    print(f"\n🔍 COMPREHENSIVE INTERPRETATION OF Gi* RESULTS")
    print("=" * 70)
    
    # Overall clustering assessment
    total_pixels = len(all_values)
    sig_hot = np.sum(np.array(all_values) > 1.96)
    sig_cold = np.sum(np.array(all_values) < -1.96)
    sig_total = sig_hot + sig_cold
    
    print(f"\n1. SPATIAL CLUSTERING OVERVIEW:")
    print(f"   • Total pixels analyzed: {total_pixels:,}")
    print(f"   • Significant clustering: {sig_total:,} pixels ({sig_total/total_pixels*100:.1f}%)")
    print(f"   • Hot spots (economic centers): {sig_hot:,} pixels ({sig_hot/total_pixels*100:.1f}%)")
    print(f"   • Cold spots (rural/underdeveloped): {sig_cold:,} pixels ({sig_cold/total_pixels*100:.1f}%)")
    
    # Temporal stability
    monthly_means = [monthly_stats[m]['mean'] for m in sorted(monthly_stats.keys())]
    temporal_stability = np.std(monthly_means)
    
    print(f"\n2. TEMPORAL ANALYSIS:")
    print(f"   • Mean Gi* range: {np.min(monthly_means):.3f} to {np.max(monthly_means):.3f}")
    print(f"   • Temporal stability (std): {temporal_stability:.3f}")
    
    if temporal_stability < 0.1:
        print(f"   • Assessment: HIGH temporal stability - consistent spatial patterns")
    elif temporal_stability < 0.2:
        print(f"   • Assessment: MODERATE temporal stability - some seasonal variation")
    else:
        print(f"   • Assessment: LOW temporal stability - significant temporal changes")
    
    # Distribution analysis
    overall_mean = np.mean(all_values)
    overall_std = np.std(all_values)
    skewness = np.mean(((np.array(all_values) - overall_mean) / overall_std) ** 3)
    
    print(f"\n3. DISTRIBUTION CHARACTERISTICS:")
    print(f"   • Overall mean: {overall_mean:.3f}")
    print(f"   • Standard deviation: {overall_std:.3f}")
    print(f"   • Skewness: {skewness:.3f}")
    
    if abs(skewness) < 0.5:
        distribution_desc = "approximately symmetric"
    elif skewness > 0.5:
        distribution_desc = "right-skewed (more hot spots)"
    else:
        distribution_desc = "left-skewed (more cold spots)"
    
    print(f"   • Distribution: {distribution_desc}")
    
    # Urban development implications
    print(f"\n4. URBAN DEVELOPMENT IMPLICATIONS:")
    
    high_hot_months = [m for m in monthly_stats.keys() 
                      if (monthly_stats[m]['hot_spots_95']/monthly_stats[m]['count']*100) > 5]
    
    if len(high_hot_months) > 6:
        print(f"   • Strong urban clustering detected in {len(high_hot_months)} months")
        print(f"   • Indicates well-defined economic centers in MIGEDC region")
    else:
        print(f"   • Moderate urban clustering in {len(high_hot_months)} months")
        print(f"   • May indicate dispersed development patterns")
    
    # Quality assessment
    print(f"\n5. ANALYSIS QUALITY ASSESSMENT:")
    
    extreme_values = np.sum((np.array(all_values) > 5) | (np.array(all_values) < -5))
    if extreme_values > 0:
        print(f"   ⚠️ {extreme_values} extreme values detected (|Gi*| > 5)")
        print(f"     This may indicate outliers or processing artifacts")
    else:
        print(f"   ✅ No extreme values detected - good data quality")
    
    # Recommendations
    print(f"\n6. RECOMMENDATIONS FOR FURTHER ANALYSIS:")
    print(f"   • Investigate hot spot locations for urban planning insights")
    print(f"   • Examine cold spots for rural development opportunities")
    print(f"   • Consider seasonal factors if temporal variation is high")
    print(f"   • Validate results with ground truth data or satellite imagery")
    
    return {
        'total_pixels': total_pixels,
        'significant_clustering_pct': sig_total/total_pixels*100,
        'hot_spots_pct': sig_hot/total_pixels*100,
        'cold_spots_pct': sig_cold/total_pixels*100,
        'temporal_stability': temporal_stability,
        'distribution_skewness': skewness,
        'extreme_values': extreme_values
    }

def generate_sample_plot():
    """Generate a sample matplotlib plot showing the color schemes."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Getis-Ord Gi* Analysis - Color Schemes', fontsize=16, fontweight='bold')
    
    # Sample data
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    
    # 1. Gi* values
    gi_star_data = X * np.exp(-(X**2 + Y**2)/2)
    im1 = axes[0,0].imshow(gi_star_data, extent=[-3,3,-3,3], cmap='RdBu_r', vmin=-3, vmax=3)
    axes[0,0].set_title('Gi* Statistics\n(Red = Hot Spots, Blue = Cold Spots)')
    plt.colorbar(im1, ax=axes[0,0], label='Gi* Value')
    
    # 2. Z-scores
    z_score_data = 2 * gi_star_data
    im2 = axes[0,1].imshow(z_score_data, extent=[-3,3,-3,3], cmap='RdBu_r', vmin=-4, vmax=4)
    axes[0,1].set_title('Z-Scores\n(Statistical Significance)')
    plt.colorbar(im2, ax=axes[0,1], label='Z-Score')
    
    # 3. Significance classification
    sig_data = np.zeros_like(gi_star_data)
    sig_data[z_score_data > 2.58] = 2
    sig_data[(z_score_data > 1.96) & (z_score_data <= 2.58)] = 1
    sig_data[(z_score_data >= -1.96) & (z_score_data <= 1.96)] = 0
    sig_data[(z_score_data < -1.96) & (z_score_data >= -2.58)] = -1
    sig_data[z_score_data < -2.58] = -2
    
    colors = ['#08306b', '#6baed6', '#f7f7f7', '#fc8d59', '#b30000']
    cmap = ListedColormap(colors)
    bounds = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
    norm = BoundaryNorm(bounds, cmap.N)
    
    im3 = axes[1,0].imshow(sig_data, extent=[-3,3,-3,3], cmap=cmap, norm=norm)
    axes[1,0].set_title('Significance Classification\n(-2: Very Cold, 2: Very Hot)')
    
    # Create custom legend
    legend_elements = [
        mpatches.Patch(color='#08306b', label='Very Significant Cold (-2)'),
        mpatches.Patch(color='#6baed6', label='Significant Cold (-1)'),
        mpatches.Patch(color='#f7f7f7', label='Not Significant (0)'),
        mpatches.Patch(color='#fc8d59', label='Significant Hot (1)'),
        mpatches.Patch(color='#b30000', label='Very Significant Hot (2)')
    ]
    axes[1,0].legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    
    # 4. NTL change example
    ntl_change_data = np.random.normal(0, 1, (100, 100))
    im4 = axes[1,1].imshow(ntl_change_data, extent=[-3,3,-3,3], cmap='viridis', vmin=-3, vmax=3)
    axes[1,1].set_title('Nighttime Lights Change\n(Sample - Baseline)')
    plt.colorbar(im4, ax=axes[1,1], label='NTL Change')
    
    plt.tight_layout()
    plt.savefig('getis_ord_color_schemes.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nSample visualization saved as 'getis_ord_color_schemes.png'")

def main():
    """Main function to run visualization utilities."""
    
    print("Getis-Ord Gi* Analysis - Visualization Utilities")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    
    # Try to load actual Gi* data first
    gi_star_data = load_gi_star_data("./data")
    
    if gi_star_data:
        print("\n🎯 Using actual Gi* results from GeoTIFF files!")
        
        # Analyze the loaded data
        monthly_stats, all_values = analyze_gi_star_statistics(gi_star_data)
        
        # Create comprehensive visualizations
        create_monthly_comparison_plot(gi_star_data, monthly_stats)
        create_spatial_maps(gi_star_data, months_to_plot=[1, 4, 7, 10])
        create_significance_summary(monthly_stats)
        interpret_gi_star_results(monthly_stats, all_values)
        
        print(f"\n✅ Real data analysis complete!")
        print(f"   📊 {len(gi_star_data)} months analyzed")
        print(f"   🗺️ Spatial maps created")
        print(f"   📈 Statistical summaries generated")
        print(f"   🔍 Interpretation provided")
        
    else:
        print("\n⚠️ No actual data found - showing visualization parameters")
        
        # Show visualization parameters
        visualize_results(config)
        
        # Generate sample plots
        print(f"\nGenerating sample visualization plots...")
        generate_sample_plot()
        
        print("\nNext steps:")
        print("1. Run simple_analysis.py to generate Gi* results")
        print("2. Re-run this script to analyze actual results")
        print("3. Use the generated plots for your research")

if __name__ == "__main__":
    main()
