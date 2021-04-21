import mysql.connector

###########################  Connecting to mysql server  #####################################################

mydb = mysql.connector.connect(
    host='localhost',
    username='senthilkumar',
    password='Senthilrf123&'
)
print(mydb)

###########################  Creating to a new database  #######################################################

mycursor = mydb.cursor() # initiates a cursor. Cursors enable manipulation of whole result sets at once. \
# In this scenario, a cursor enables the sequential processing of rows in a result set. In SQL procedures, \
# a cursor makes it possible to define a result set (a set of data rows) and perform complex logic on a row by \
# row basis.
mycursor.execute('CREATE DATABASE bikes')

##########################  Showing databases created so far  #####################################################

mycursor.execute('SHOW DATABASES')
for database in mycursor:
    print(database)

########################################  Create tabel in database bikes  #################################

mydb = mysql.connector.connect(
  host="localhost",
  user="senthilkumar",
  password="Senthilrf123&",
  database="bikes"
)
mycursor = mydb.cursor()
mycursor.execute("CREATE TABLE customers (name VARCHAR(255), address VARCHAR(255))") # Note: Can't create a table
#without a column

# to check if table exists or not
mycursor.execute("SHOW TABLES")
for table in mycursor:
    print(table)

####################################  Primery key  ############################################

# When creating a table, you should also create a column with a unique key for each record.This can be done by \
#defining a PRIMARY KEY. We use the statement "INT AUTO_INCREMENT PRIMARY KEY" which will insert a unique number\
# for each record. Starting at 1, and increased by one for each record.

mycursor.execute("CREATE TABLE customers1 (id INT AUTO_INCREMENT PRIMARY KEY, name VARCHAR(255), address VARCHAR(255));")
#adding a colomn in table
mycursor.execute("ALTER TABLE customers ADD COLUMN id INT AUTO_INCREMENT PRIMARY KEY")

#######################  show the table records   ##################################

#1. To show all the records

mycursor.execute('SELECT * FROM customers;')
records = mycursor.fetchall()
for row in records:
    print(row)

#2. To show only one record

mycursor.execute('SELECT * FROM customers;')
records = mycursor.fetchone()
for row in records:
    print(row)

#3. To limit the search records to a specific count

mycursor.execute("SELECT * FROM customers LIMIT 5") #You can limit the number of records returned from the query,\
# by using the "LIMIT" statement:

myresult = mycursor.fetchall()
for x in myresult:
  print(x)

#4. To skip few records to show the search results

mycursor.execute("SELECT * FROM customers LIMIT 5 OFFSET 2") #If you want to return five records, starting from the \
# third record, you can use the "OFFSET" keyword

myresult = mycursor.fetchall()
for x in myresult:
  print(x)

############################  To insert a record to the table  #######################################################

sql = "INSERT INTO customers (name, address) VALUES (%s, %s)"
val = ("John", "Highway 21")
mycursor.execute(sql, val)
mydb.commit()
print(mycursor.rowcount, "record inserted.")

# To insert multiple record to the table

sql = "INSERT INTO customers (name, address) VALUES (%s, %s)"
val = [
  ('Peter', 'Lowstreet 4'),
  ('Amy', 'Apple st 652'),
  ('Hannah', 'Mountain 21'),
  ('Michael', 'Valley 345'),
  ('Sandy', 'Ocean blvd 2'),
  ('Betty', 'Green Grass 1'),
  ('Richard', 'Sky st 331'),
  ('Susan', 'One way 98'),
  ('Vicky', 'Yellow Garden 2'),
  ('Ben', 'Park Lane 38'),
  ('William', 'Central st 954'),
  ('Chuck', 'Main Road 989'),
  ('Viola', 'Sideway 1633')
]
mycursor.executemany(sql, val) #execute many is used here
mydb.commit()
print(mycursor.rowcount, "was inserted.")

#Getting Id of last inserted row:

sql = "INSERT INTO customers (name, address) VALUES (%s, %s)"
val = ("Michelle", "Blue Village")
mycursor.execute(sql, val)
mydb.commit()
print("1 record inserted, ID:", mycursor.lastrowid)

# Filter the selection with WHERE

sql = "SELECT * FROM customers WHERE address ='Park Lane 38'"
mycursor.execute(sql)
myresult = mycursor.fetchall()
for x in myresult:
  print(x)

# where with wildcard statement

sql = "SELECT * FROM customers WHERE address LIKE '%way%'"

mycursor.execute(sql)
myresult = mycursor.fetchall()
for x in myresult:
  print(x)

# TO keep the  hackers away and when the query value is provide by user, you should escape the values. Escape query \
# values by using the placholder %s method:

sql = "SELECT * FROM customers WHERE address = %s"
adr = ("Yellow Garden 2", )
mycursor.execute(sql, adr)
myresult = mycursor.fetchall()
for x in myresult:
  print(x)

####################################   Sort the results   ####################################################

#Use the ORDER BY statement to sort the result in ascending or descending order.

#The ORDER BY keyword sorts the result ascending by default. To sort the result in descending order, use the DESC keyword.
sql = "SELECT * FROM customers ORDER BY name"
sql_desc = "SELECT * FROM customers ORDER BY name DESC"

mycursor.execute(sql_desc)
myresult = mycursor.fetchall()
for x in myresult:
    print(x)

#######################################    Deleting records   ###########################################

#You can delete records from an existing table by using the "DELETE FROM" statement:

sql = "DELETE FROM customers WHERE address = 'Mountain 21'"
mycursor.execute(sql)
mydb.commit()
print(mycursor.rowcount, "record(s) deleted")

# Avoid SQL injection while deleting

sql = "DELETE FROM customers WHERE address = %s"
adr = ("Yellow Garden 2", )
mycursor.execute(sql, adr)
mydb.commit()
print(mycursor.rowcount, "record(s) deleted")

#######################  To delete a table  ########################################

#You can delete an existing table by using the "DROP TABLE" statement:

sql = "DROP TABLE customers1"
mycursor.execute(sql)

# or delete only if it table exists

sql = "DROP TABLE IF EXISTS customers1"
mycursor.execute(sql)

######################  Update a table  #######################################

#You can update existing records in a table by using the "UPDATE" statement

sql = "UPDATE customers SET address = 'Canyon 123' WHERE address = 'Valley 345'"
mycursor.execute(sql)
mydb.commit()
print(mycursor.rowcount, "record(s) affected")

#Notice the WHERE clause in the UPDATE syntax: The WHERE clause specifies which record or records \
# that should be updated. If you omit the WHERE clause, all records will be updated!

# to avoid sql injection, during update

sql = "UPDATE customers SET address = %s WHERE address = %s"
val = ("Valley 345", "Canyon 123")
mycursor.execute(sql, val)
mydb.commit()
print(mycursor.rowcount, "record(s) affected")

##########################  Join tables   ##################################################

#You can combine rows from two or more tables, based on a related column between them, by using a JOIN statement.

#Consider you have a "users" table and a "products" table:

#creating table users
mycursor.execute("CREATE TABLE users (id VARCHAR(255), name VARCHAR(255),fav VARCHAR(255))") # Note: Can't create a table
sql = "INSERT INTO users (id, name, fav) VALUES (%s, %s, %s)"
val = [
  ('1','John', '154'),
  ('2','Peter', '154'),
('3','Amy', '155'),
('4','Hannah', ''),
('5','Michael', '')
]
mycursor.executemany(sql, val)   #execute many is used
mydb.commit()

#creating table products

mycursor.execute("CREATE TABLE products (id VARCHAR(255),name VARCHAR(255))") # Note: Can't create a table
sql = "INSERT INTO products (id, name) VALUES ( %s, %s)"
val = [
  ('154', 'Chocolate Heaven'),
  ('155', 'Tasty Lemons'),
('156', 'Vanilla Dreams')
]
mycursor.executemany(sql, val)   #execute many is used
mydb.commit()


mycursor.execute('SELECT * FROM users')
records = mycursor.fetchall()
for row in records:
    print(row)

#1. INNER JOIN

sql = "SELECT \
  users.name AS user, \
  products.name AS favorite \
  FROM users \
  INNER JOIN products ON users.fav = products.id"

mycursor.execute(sql)

myresult = mycursor.fetchall()

for x in myresult:
  print(x)

#result
"""
('John', 'Chocolate Heaven')
('Peter', 'Chocolate Heaven')
('Amy', 'Tasty Lemons')
"""
#In the example above, Hannah, and Michael were excluded from the result, that is because INNER JOIN only shows the \
# records where there is a match.
#Note: You can use JOIN instead of INNER JOIN. They will both give you the same result.

#3. LEFT JOIN(fill values from right table and keeps all records in left table)
#you want to show all users, even if they do not have a favorite product, use the LEFT JOIN statement

sql = "SELECT \
  users.name AS user, \
  products.name AS favorite \
  FROM users \
  LEFT JOIN products ON users.fav = products.id"

mycursor.execute(sql)

myresult = mycursor.fetchall()

for x in myresult:
  print(x)

#Results

"""
('John', 'Chocolate Heaven')
('Peter', 'Chocolate Heaven')
('Amy', 'Tasty Lemons')
('Hannah', None)
('Michael', None)
"""

#3. RIGHT JOIN(fill the values from left to all record of right table)

#If you want to return all products, and the users who have them as their favorite, even if no user have them as\
# their favorite, use the RIGHT JOIN statement:

sql = "SELECT \
  users.name AS user, \
  products.name AS favorite \
  FROM users \
  RIGHT JOIN products ON users.fav = products.id"

mycursor.execute(sql)

myresult = mycursor.fetchall()

for x in myresult:
  print(x)

# RESULT

"""
('Peter', 'Chocolate Heaven')
('John', 'Chocolate Heaven')
('Amy', 'Tasty Lemons')
(None, 'Vanilla Dreams')

#Note: Hannah and Michael, who have no favorite product, are not included in the result.
"""