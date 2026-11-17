"""Optional Sentinel-2 L2A NDVI-input extraction to a Google Drive CSV.

STATUS: written against the Earth Engine Python API but NOT executed by the project
authors' tooling (no credentials or study-area asset were available). Run it on one
small field first and inspect the CSV before trusting it.

Prerequisites: an Earth Engine-enabled Google Cloud project, an authenticated
environment (`earthengine authenticate`), and a FeatureCollection asset whose
features have a `field_id` property. Example:

  python scripts/export_sentinel2_gee.py --project YOUR_GCP_PROJECT \
    --asset projects/YOUR_GCP_PROJECT/assets/wheat_fields \
    --start 2022-11-01 --end 2023-05-01 --season 2022-23 --drive-folder PT355_exports

One row is exported per field per Sentinel-2 scene with the columns the app expects:
field_id, season, observation_date, red_reflectance (B4), nir_reflectance (B8).
Values are surface reflectance scaled to 0-1 (GEE stores 10000x). Fields fully masked
by cloud/shadow in a scene are exported with empty reflectance and are counted as
invalid by the pipeline's data-quality report.
Do not commit tokens or service-account keys; authentication stays in your environment.
"""
import argparse

import ee

SCL_MASKED = (3, 8, 9, 10, 11)  # cloud shadow, cloud medium/high prob., thin cirrus, snow/ice
SCALE_FACTOR = 0.0001  # Sentinel-2 L2A reflectance is stored as integers x 10000


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--asset", required=True, help="Earth Engine FeatureCollection asset ID (needs a field_id property)")
    parser.add_argument("--project", required=True, help="Google Cloud project registered for Earth Engine")
    parser.add_argument("--start", required=True, help="Inclusive start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="Exclusive end date YYYY-MM-DD")
    parser.add_argument("--season", required=True, help="Season label; must match the yield/weather CSV exactly")
    parser.add_argument("--drive-folder", default="PT355_exports")
    parser.add_argument("--cloud-percent", type=float, default=80, help="Scene-level CLOUDY_PIXEL_PERCENTAGE ceiling")
    parser.add_argument("--scale", type=int, default=10, help="Reduction scale in metres (B4/B8 are 10 m)")
    args = parser.parse_args()

    ee.Initialize(project=args.project)
    fields = ee.FeatureCollection(args.asset)

    def mask_clouds(image):
        scl = image.select("SCL")
        good = ee.Image(1)
        for cls in SCL_MASKED:
            good = good.And(scl.neq(cls))
        return image.updateMask(good).copyProperties(image, ["system:time_start"])

    collection = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                  .filterDate(args.start, args.end)
                  .filterBounds(fields.geometry())
                  .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", args.cloud_percent))
                  .map(mask_clouds))

    def per_image(image):
        date = ee.Date(image.get("system:time_start")).format("YYYY-MM-dd")
        reduced = image.select(["B4", "B8"]).reduceRegions(collection=fields, reducer=ee.Reducer.mean(), scale=args.scale)

        def tag(feature):
            red, nir = ee.Number(feature.get("B4")), ee.Number(feature.get("B8"))
            return ee.Feature(None, {
                "field_id": feature.get("field_id"), "season": args.season, "observation_date": date,
                "red_reflectance": red.multiply(SCALE_FACTOR), "nir_reflectance": nir.multiply(SCALE_FACTOR)})
        return reduced.map(tag)

    table = collection.map(per_image).flatten()
    task = ee.batch.Export.table.toDrive(
        collection=table, description="PT355_Sentinel2_" + args.season.replace("/", "-"), folder=args.drive_folder,
        fileFormat="CSV", selectors=["field_id", "season", "observation_date", "red_reflectance", "nir_reflectance"])
    task.start()
    print("Earth Engine export task started:", task.id)
    print("Monitor it at https://code.earthengine.google.com/tasks and download the CSV from Drive.")


if __name__ == "__main__":
    main()
