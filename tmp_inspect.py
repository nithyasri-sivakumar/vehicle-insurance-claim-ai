import sqlite3
from pathlib import Path
p = Path('c:/Users/nithy/OneDrive/Desktop/AI vehicle insurance claim/insurance.db')
print('DB exists:', p.exists(), p)
conn = sqlite3.connect(str(p))
c = conn.cursor()
c.execute('PRAGMA table_info(claim)')
print(c.fetchall())
conn.close()