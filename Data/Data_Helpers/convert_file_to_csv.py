import shapefile
import csv

shape = shapefile.Reader(r"/mnt/c/Users/david/Downloads/Claude Projects/Geocoding ML/Data/us-ct-city_of_stamford/Address.shp")

fields = shape.fields

#wslpath -u "C:\Users\david\Downloads\Claude Projects\Geocoding ML\Data\us-ct-city_of_stamford\Address.shp"


with open("output.csv", "w", newline = "", encoding ="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames = [field.name for field in fields])
    writer.writeheader()

    writer.writerows(record.as_dict() for record in shape.records())

print(fields)