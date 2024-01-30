import csv

sutdent_names_file = '/home/zoltan/Documents/database/students.csv'

# psql -h webgis.fmt.bme.hu -p 25432 -U postgres sslmode=verify-full

print('CREATE ROLE student;')

with open(sutdent_names_file, newline='') as csvfile:
    header = next(csvfile)
    rows = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in rows:
        neptun = row[1].lower()
        
        query = '\n'
        
        # query += "CREATE USER " + neptun + " with encrypted password '123456';\n"
        # query += 'GRANT student TO ' + neptun + ';\n'
        # query += 'CREATE DATABASE ' + neptun + ';\n'
        # query += 'REVOKE CONNECT ON DATABASE ' + neptun + ' FROM student;\n' 
        # query += 'REVOKE CONNECT ON DATABASE ' + neptun + ' FROM PUBLIC;\n'
        # query += 'GRANT ALL PRIVILEGES ON DATABASE ' + neptun + ' to ' + neptun + ';\n'
        # query += 'ALTER USER ' + neptun + ' SET search_path = ' + neptun + ';\n'

        print(query)
        #break
        # db_name = row[0].replace('"', '')
        # if len(db_name) == 6:
        #     print('DROP DATABASE IF EXISTS ' + db_name)