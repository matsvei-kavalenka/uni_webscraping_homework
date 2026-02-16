import csv


def write_csv(path, data):
    fieldnames = data[0].keys()
    with open(path, 'w', newline='', encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
