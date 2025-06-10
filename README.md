# Getis-Ord Gi* Analysis of Nighttime Lights

This project performs spatial autocorrelation analysis using Getis-Ord Gi* statistics on VIIRS nighttime lights data for MIGEDC (Metro Iloilo-Guimaras Economic Development Council), with water masking to remove outliers and avoid extreme z-scores.

## Features

- **Water Masking**: Uses JRC Global Surface Water dataset to exclude water bodies from analysis
- **Quality Filtering**: Applies quality and snow masks to ensure data reliability
- **Outlier Removal**: Removes extreme values using percentile thresholds
- **Spatial Autocorrelation**: Implements Getis-Ord Gi* statistics to identify hot and cold spots
- **Statistical Significance**: Calculates z-scores and classifies significance levels
- **Automated Export**: Exports results to Google Drive for further analysis

## Setup

### Quick Setup (Windows)
1. **Run the setup batch file**:
   ```powershell
   setup.bat
   ```

### Manual Setup
1. **Install Dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

2. **Run Setup Script**:
   ```powershell
   python setup.py
   ```

3. **Authenticate Google Earth Engine**:
   ```powershell
   earthengine authenticate
   ```

4. **Configure Analysis Parameters**:
   Edit `config.yaml` to set your analysis parameters:
   - Project ID
   - Region of interest (ROI)
   - Date ranges for baseline and sample periods
   - Processing months
   - Export settings

## Configuration

The `config.yaml` file contains all analysis parameters:

```yaml
ee:
  project: your-gee-project-id
  roi: [122.0, 10.0, 123.0, 11.0]  # [west, south, east, north]
  ntl_collection_id: NASA/VIIRS/002/VNP46A2
  ntl_band: Gap_Filled_DNB_BRDF_Corrected_NTL
  quality_band: Mandatory_Quality_Flag
  snow_band: Snow_Flag

baseline:
  start_year: 2015
  end_year: 2019

sample:
  start_year: 2024
  end_year: 2024

process_months: [1,2,3,4,5,6,7,8,9,10,11,12]  # All months
export:
  output_dir: "./data"
  scale: 500  # Resolution in meters
```

## Usage

Run the analysis script:

```powershell
python getis_ord_analysis.py
```

## Output Files

The script exports several files to Google Drive:

1. **baseline_ntl_{years}**: Mean nighttime lights for baseline period
2. **sample_ntl_{years}**: Mean nighttime lights for sample period
3. **ntl_change_{periods}**: Change in nighttime lights (sample - baseline)
4. **getis_ord_gi_star_{year}**: Gi* statistics values
5. **getis_ord_z_score_{year}**: Z-scores for statistical significance
6. **getis_ord_significance_{year}**: Classified significance levels
7. **getis_ord_complete_analysis_{year}**: Combined results

## Interpretation

### Gi* Values
- **Positive values**: Local clustering of high values (hot spots)
- **Negative values**: Local clustering of low values (cold spots)
- **Values near zero**: No significant spatial clustering

### Z-Scores and Significance
- **|Z-score| > 1.96**: Statistically significant at p < 0.05 (95% confidence)
- **|Z-score| > 2.58**: Statistically significant at p < 0.01 (99% confidence)

### Significance Classification
- **2**: Very significant hot spots (z > 2.58)
- **1**: Significant hot spots (1.96 < z ≤ 2.58)
- **0**: Not significant (-1.96 ≤ z ≤ 1.96)
- **-1**: Significant cold spots (-2.58 ≤ z < -1.96)
- **-2**: Very significant cold spots (z < -2.58)

## Analysis Workflow

1. **Data Preparation**:
   - Load VIIRS nighttime lights data
   - Apply quality and snow masks
   - Create water mask using JRC Global Surface Water

2. **Temporal Analysis**:
   - Compute mean nighttime lights for baseline period
   - Compute mean nighttime lights for sample period
   - Calculate change (sample - baseline)

3. **Outlier Removal**:
   - Remove extreme values using 5th and 95th percentiles
   - Apply water mask to exclude water bodies

4. **Spatial Analysis**:
   - Compute Getis-Ord Gi* statistics
   - Calculate z-scores for statistical significance
   - Classify significance levels

5. **Export Results**:
   - Export all analysis results to Google Drive
   - Generate comprehensive statistics

## Technical Details

### Water Masking
- Uses JRC Global Surface Water dataset
- Excludes pixels with >50% water occurrence
- Helps remove extreme z-scores from water bodies

### Getis-Ord Gi* Implementation
- Uses 3x3 pixel neighborhood for local statistics
- Computes standardized Gi* values
- Calculates z-scores for hypothesis testing

### Quality Control
- Multiple masking layers (quality, snow, water)
- Percentile-based outlier removal
- Statistical validation of results

## Troubleshooting

1. **Authentication Issues**: Run `earthengine authenticate` and follow prompts
2. **Memory Errors**: Reduce ROI size or increase scale parameter
3. **Export Failures**: Check Google Drive permissions and storage space
4. **Data Gaps**: Verify date ranges and collection availability

## References

- Getis, A., & Ord, J. K. (1992). The analysis of spatial association by use of distance statistics
- Ord, J. K., & Getis, A. (1995). Local spatial autocorrelation statistics
- VIIRS Nighttime Lights: https://developers.google.com/earth-engine/datasets/catalog/NOAA_VIIRS_DNB_MONTHLY_V1_VCMSLCFG
