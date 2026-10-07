# Figure Captions

fig01_missingness.png  
Share of missing values and -999 sentinels in every column of the three raw tables.  
**Takeaway:** Plot data has a few sentinel-coded columns (esp. pest_disease_flag); weather and price tables are nearly complete but still need standardisation.

fig02_before_after_cleaning.png  
Distributions of farm size, fertilizer, price and pest flag before vs after cleaning.  
**Takeaway:** Cleaning removed extreme outliers and unit errors; the price unit mix-up is clearly visible and fixed.

fig03_yield_distribution.png  
Overall histogram and crop-wise violin plots of yield_tons_per_ha.  
**Takeaway:** Yield is right-skewed; maize and wheat sit higher than teff/sorghum/barley on average.

fig04_region_crop_heatmap.png  
Annotated heatmap of mean yield by region × crop (cell = mean + n).  
**Takeaway:** Best combinations are region-specific (e.g. Oromia maize); some cells have low n and should be interpreted cautiously.

fig05_correlation_heatmap.png  
Pearson correlations among yield, numeric plot features and weather-derived features.  
**Takeaway:** Soil quality, fertilizer and season temperature show the strongest linear associations with yield.

fig06_climate_by_region.png  
Monthly average temperature and rainfall by region with an example growing-season window shaded.  
**Takeaway:** Regions differ markedly in temperature and rainfall seasonality; the growing-season window must be chosen carefully for the weather join.

fig07_yield_vs_season_temp.png  
Scatter + binned means of yield against growing-season mean temperature, one panel per crop.  
**Takeaway:** Most crops show a mild optimum or gentle decline at higher temperatures; the relationship is crop-specific.

fig08_price_trends.png  
Average (unit-corrected) price per quintal by year for each crop.  
**Takeaway:** Prices rose for every crop 2021–2024; the relative ranking of crops stayed fairly stable.

fig09_revenue_by_crop_region.png  
Grouped bars of mean revenue (birr/ha) by crop and region.  
**Takeaway:** High-yield crops in high-price regions dominate revenue; regional price differences amplify yield gaps.

fig10_model_comparison.png  
Validation RMSE for every model tried, baseline marked, CV error bars on final models.  
**Takeaway:** HistGradientBoosting is the clear winner; the mean predictor is far behind.

fig11_predicted_vs_actual_residuals.png  
Predicted-vs-actual scatter (y=x line) and residual-vs-predicted plot.  
**Takeaway:** Predictions track the diagonal well; residuals are roughly homoscedastic with a few large outliers.

fig12_feature_importance.png  
Top 15 features by importance; weather-derived features highlighted.  
**Takeaway:** Soil quality, fertilizer and season temperature dominate; several weather features rank in the top 15, justifying the join.
