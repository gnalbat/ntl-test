@echo off
echo Getis-Ord Gi* Analysis - Windows Setup
echo =====================================

echo.
echo Installing required packages...
pip install -r requirements.txt

echo.
echo Running setup script...
python setup.py

echo.
echo Setup complete! 
echo.
echo Next steps:
echo 1. Review config.yaml
echo 2. Run: python getis_ord_analysis.py
echo 3. Visualize: python visualize_results.py

pause
