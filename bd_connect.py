import psycopg2

conn = psycopg2.connect(database = "diario_republica", 
                        user = "postgres", 
                        host= 'localhost',
                        password = "1597535Omg.",
                        port = 5432)

cur = conn.cursor()

cur.execute("SELECT * FROM diario_republica")

records = cur.fetchall()