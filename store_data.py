import sqlite3
import pandas as pd

df = pd.read_csv("data.csv")

# print(df.head())

connection = sqlite3.connect('ticket.db')

df.to_sql('user_ticket',connection, if_exists='replace')

# connection.close()

cursor = connection.cursor()

for row in cursor.execute("SELECT * FROM user_ticket"):
    print(row)

connection.close()


