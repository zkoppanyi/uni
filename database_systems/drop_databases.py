import csv

# 1. Query databases: SELECT datname FROM pg_catalog.pg_database;
# 2. Save the query into a CSV file
# 3. Specify the CSV file here:
database_names_file = '/home/zoltan/Downloads/data-1693578323927.csv'

# psql -h webgis.fmt.bme.hu -p 25432 -U postgres sslmode=verify-full

with open(database_names_file, newline='') as csvfile:
    rows = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in rows:
        db_name = row[0].replace('"', '')
        if len(db_name) == 6:
            print('DROP DATABASE IF EXISTS ' + db_name)