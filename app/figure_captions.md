# Figure captions

Each caption says what the figure shows and the takeaway. Figures 10-12 use a quick model trained on the train file only (see notebook 03); the full modeling write-up is Deliverable D.

**fig01_missingness.png** - Share of empty cells and -999 codes in each of the 25 raw columns across the plot, weather and price tables. 8 columns are affected (worst: fertilizer_kg_per_ha, 5.0% of rows), so no column is badly broken, but the -999 codes in particular would silently distort averages if left in.

**fig02_before_after_cleaning.png** - Box plots of raw versus cleaned values: farm size reached 60 ha and fertilizer 899 kg/ha, labour contained a -999 code, and some prices were 34 birr (per kg, not per quintal). After capping, NaN handling and unit repair the ranges are up to 5 ha, 163 kg/ha, 5-105 labour days and 2,403-8,811 birr, so the bulk of the data is now readable.

**fig03_yield_distribution.png** - Yield is right-skewed overall (mean 2.76 vs median 2.58 t/ha), and the crop violins show maize clearly highest (median 3.77) and teff lowest (2.00). Crop type clearly shifts the whole distribution, and maize and wheat are the most variable crops (SD 1.58 and 1.41 t/ha).

**fig04_region_crop_heatmap.png** - Mean yield for every region and crop pair, with plot counts (533 to 645 per cell, all large enough to trust). SNNPR maize is best (4.79 t/ha) and Somali teff worst (0.83); Somali is lowest for every crop, so both the region and the crop-region pairing matter.

**fig05_correlation_heatmap.png** - Pairwise correlations between yield, the plot features and the weather-derived features (orange labels). The weather features correlate most strongly with yield (extreme-heat days -0.46, season temperature -0.45, station rainfall +0.30) while distance to market (+0.01) and plot-reported rainfall (+0.03) show almost none; the weather features also overlap heavily with each other, partly because the hot Somali region drives all of them.

**fig06_climate_by_region.png** - Monthly mean temperature and rainfall for each region, with the growing seasons for a February (Feb-May) and a June (Jun-Sep) planting shaded. Somali is hot (28.7 C) and dry (30 mm per month) while the others average 17.1-20.5 C, and every region's wettest month falls inside the Jun-Sep window, so planting month changes how much of the rains a crop catches.

**fig07_yield_vs_season_temp.png** - Plot-level yield against growing-season mean temperature for each crop, with binned means and 95% intervals. Crops do have different sweet spots: maize peaks near 21 C (5.1 t/ha) and sorghum near 20 C, while barley, wheat and teff do best in the coolest seasons (about 16-18 C) and fall steadily as it gets hotter; temperature is strongly tied to region here, so this is an association, not a controlled test.

**fig08_price_trends.png** - Average price per quintal for each crop from 2021 to 2024 (left) and indexed to 2021 = 100 (right). Every crop rose by more than 20% (from sorghum +29.9% to teff +20.8%), and teff stays the highest-priced crop in every year, so the growth gap changes budgets but not the ranking.

**fig09_revenue_by_crop_region.png** - Mean revenue per hectare (yield x 10 x price) for each crop in each region, bars starting at zero. The most profitable crop is teff in Oromia, SNNPR and Tigray, wheat in Amhara and maize in Somali, and Somali revenues (51-68 thousand birr/ha) are far below the other regions (78-176), so the best crop depends on where the farm is.

**fig10_model_comparison.png** - Validation RMSE of each model (bars) with 5-fold cross-validation mean and spread (dots) and the mean-predictor baseline (dashed line). Gradient boosting (HistGB) is best at 0.490 t/ha versus 1.401 for the baseline (65% lower), ahead of the random forest (0.536) and ridge (0.886), and its small CV spread (+/-0.006) shows the ranking is stable.

**fig11_predicted_vs_actual_residuals.png** - Left: predicted versus actual yield on the held-out 20% with the y = x line; right: residuals versus predicted yield. Predictions track the line well (R2 0.88, RMSE 0.49, MAE 0.35 t/ha) with no overall bias (mean residual -0.01), but the error grows with predicted yield (residual SD 0.25 in the lowest decile to 0.81 in the highest), so high-yield plots are the least certain.

**fig12_feature_importance.png** - Permutation importance (increase in validation RMSE when a feature is shuffled) for the top 15 features, weather-derived features in orange. Crop type (0.95), altitude (0.75) and season mean temperature (0.48, the top weather feature) lead, weather-derived features together carry 17% of total importance, and plot-reported rainfall adds only 0.03.
