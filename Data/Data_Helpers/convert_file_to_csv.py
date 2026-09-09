import os

import shapefile
import csv

shape = shapefile.Reader("../us-ct-city_of_stamford/Address.shp")

fields = shape.fields

#wslpath -u "C:\Users\david\Downloads\Claude Projects\Geocoding ML\Data\us-ct-city_of_stamford\Address.shp"

if not os.path.exists("output.csv"):
    with open("output.csv", "w", newline = "", encoding ="utf-8") as f:
        ignore_cols = ["DeletionFlag"]
        writer = csv.DictWriter(f, fieldnames = [field.name for field in fields if field.name not in ignore_cols])
        writer.writeheader()

        writer.writerows(record.as_dict() for record in shape.records(fields= [f.name for f in fields if f.name not in ignore_cols]))

print(fields)