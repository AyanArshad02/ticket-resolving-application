import sqlite3
import pandas as pd
import os

# Define file paths
DATA_PATH = "backend/data/data.csv"
DB_PATH = "backend/ticket.db"

# Load CSV into DataFrame
df = pd.read_csv(DATA_PATH)

# Connect to SQLite database
with sqlite3.connect(DB_PATH) as connection:
    # Store DataFrame into SQLite table
    df.to_sql('user_ticket', connection, if_exists='replace', index=False)

    # Fetch and print all rows for verification
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM user_ticket")
    rows = cursor.fetchall()

    for row in rows:
        print(row)










# import sqlite3
# import pandas as pd

# df = pd.read_csv("backend/data/data.csv")

# # print(df.head())

# connection = sqlite3.connect('backend/ticket.db')

# df.to_sql('user_ticket',connection, if_exists='replace')

# # connection.close()

# cursor = connection.cursor()

# for row in cursor.execute("SELECT * FROM user_ticket"):
#     print(row)

# connection.close()


