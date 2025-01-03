import sqlite3

def view_database():
    try:
        # Step 1: Connect to the SQLite database
        conn = sqlite3.connect("health_predictions.db")
        cursor = conn.cursor()
        
        # Step 2: Write the query to fetch data
        query = "SELECT * FROM Predictions"
        cursor.execute(query)
        
        # Step 3: Fetch all rows and display them
        rows = cursor.fetchall()
        if rows:
            print("Stored Data in the Database:")
            for row in rows:
                print(row)
        else:
            print("The database is empty. No data to display.")
    except sqlite3.Error as e:
        print(f"Error occurred while accessing the database: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    # Run the function to view database contents
    view_database()
