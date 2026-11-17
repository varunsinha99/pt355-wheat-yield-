# Data provenance checklist (to be completed by the team — intentionally blank)

| Question | Answer |
|---|---|
| Kaggle (or other) dataset name, URL, version, access date | |
| Licence / terms; attribution text | |
| Is the crop wheat only? How was it filtered? | |
| Yield column name and unit (t/ha, kg/ha, …) | |
| Weather variables, source, and how aggregated to one row per field-season | |
| Do records have coordinates/field polygons? CRS? | |
| Season definition (sowing–harvest dates) and label format | |
| Sentinel-2 collection, date window, cloud filter, mask classes, scale (10 m) | |
| Reflectance scale (0–1 or 0–10000) and processing baseline | |
| Known gaps, imputed values, outliers removed (and why) | |
| Any column derived from yield (production, area×yield)? | |
| Who verified field-ID and season matching between files? | |

Keep private data in `data/private/` (git-ignored). Do not publish data you are not licensed to redistribute.
