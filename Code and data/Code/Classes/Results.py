import sqlite3
import pandas as pd
from sqlite3 import Error


class Results():
    def __init__(self, path):
        self.path = path
        self.con = sqlite3.connect(self.path)

    def get_table(self,tab):
        
        c = self.con.cursor()
        c.execute(f"""
                SELECT * FROM {tab}
                """)
        columns = [d[0] for d in c.description]
        rows = [i for i in c.fetchall()]
        tab = pd.DataFrame(rows, columns = columns)
        return tab
    
    def add_table(self, df, name):

        try:
            df.to_sql(con=self.con, if_exists="fail", name=name)
        except Exception as e:  # Temporarily catching all exceptions
            if "already exists" in str(e):
                print(f"Integrity error: {e}")
           
            else:
                print(f"Exception type: {type(e).__name__}. Message: {e}")


    def overwrite_table(self, df, name):
        df.to_sql(con = self.con, if_exists = "replace", name = name)
    
    def add_model_tab(self):
        c = self.con.cursor()
        try:
            c.execute('''
            CREATE TABLE training_times (
                model TEXT NOT NULL,
                type TEXT NOT NULL,
                stop_words TEXT NOT NULL,
                dimensions INTEGER,
                time FLOAT,
                PRIMARY KEY (model, type, stop_words)
            )
            ''')
            self.con.commit()
        except Error as e:
            print(e)

        
    def add_training_time(self,row):
        """
        Insert or update the training_times table with the provided row.
        """
        c = self.con.cursor()
        
        # Upsert (insert or replace) the data
        c.execute('''
        INSERT OR REPLACE INTO training_times 
        (model, type, stop_words, dimensions, time)
        VALUES (?, ?, ?, ?, ?)
        ''', (row["model"], row["type"], row["stop_words"], row["dimensions"], row["time"]))
        
        # Commit the changes and close the connection
        self.con.commit()

    def create_kmeans_times(self):
        """
        Create the 'kmeans_times' table in SQLite.
        """
        c = self.con.cursor()
        try:
            c.execute('''
                CREATE TABLE IF NOT EXISTS kmeans_times (
                    model_name TEXT,
                    model_type TEXT,
                    a FLOAT,
                    b FLOAT,
                    c FLOAT,
                    d FLOAT,
                    PRIMARY KEY(model_name, model_type),
                    FOREIGN KEY(model_name, model_type) REFERENCES training_times(model_name, model_type)
                )
            ''')
            self.con.commit()
        except Error as e:
            print(e)


    def add_kmeans_time(self, model_name, model_type, experiment_name, time):
        """
        Update or insert a time in the 'kmeans_times' table 
        """
        c = self.con.cursor()

        ## Create the row if it doesn't exist, ignore it if it does
        c.execute(f'''
        INSERT OR IGNORE INTO kmeans_times (model_name, model_type, {experiment_name})
        VALUES (?, ?, ?)
        ''', (model_name, model_type, time))
        
        ## Update the row to put the time in the appropriate column
        c.execute(f'''
        UPDATE kmeans_times
        SET {experiment_name} = ?
        WHERE model_name = ? AND model_type = ?
        ''', (time, model_name, model_type))
        
        self.con.commit()

    def list_table_columns(self, table_name):
        """
        Retrieves and lists the columns of a specified table from the database.
        
        Args:
        - table_name (str): The name of the table to retrieve columns from.

        Returns:
        - list: A list of column names.
        """

        # Create cursor
        cursor = self.con.cursor()

        # Execute PRAGMA table_info command
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()

        # Extract the column names and return them
        return [column[1] for column in columns]

    def get_col(self, col, table, additional_sql = ""):
        c = self.con.cursor()
        c.execute(f"""
            SELECT {col} FROM {table}
            {additional_sql}
            """)
        
        return c.fetchall()
    
    def create_res_tab(self, name, value_type = "FLOAT"):
        """
        Create a results table.
        """
        c = self.con.cursor()
        try:
            c.execute(f'''
                CREATE TABLE IF NOT EXISTS {name} (
                    model TEXT,
                    type TEXT,
                    a {value_type},
                    b {value_type},
                    c {value_type},
                    d {value_type},
                    PRIMARY KEY(model, type),
                    FOREIGN KEY(model, type) REFERENCES training_times(model, type)
                )
            ''')
            self.con.commit()
        except Error as e:
            print(e)

    def add_result(self, table, model_name, model_type, experiment_name, result):
        """
        Update or insert a into a result table 
        """
        c = self.con.cursor()

        ## Create the row if it doesn't exist, ignore it if it does
        c.execute(f'''
        INSERT OR IGNORE INTO {table} (model, type, {experiment_name})
        VALUES (?, ?, ?)
        ''', (model_name, model_type, result))
        
        ## Update the row to put the time in the appropriate column
        c.execute(f'''
        UPDATE {table}
        SET {experiment_name} = ?
        WHERE model = ? AND type = ?
        ''', (result, model_name, model_type))
        
        self.con.commit()
    
    def drop_tab(self, table_name):
        c = self.con.cursor()
        c.execute(f'DROP TABLE IF EXISTS {table_name}')
        self.con.commit()

    
    def __del__(self):
        """Destructor to close the database connection."""
        if self.con:
            self.con.close()
   
if __name__ == "__main__":
    results = Results("Data/results.sqlite")
    df = results.get_table("training_times")
    df.to_csv("Data/training_times.csv")