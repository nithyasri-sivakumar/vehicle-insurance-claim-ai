from flask import Flask
from flask_login import LoginManager
from flask_wtf import CSRFProtect
import os
import sqlite3

from database.models import db, User, Claim

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change in production
db_path = os.path.join(app.root_path, 'instance', 'insurance.db')
os.makedirs(os.path.dirname(db_path), exist_ok=True)
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static', 'uploads')
app.config['WTF_CSRF_ENABLED'] = False

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db.init_app(app)
login_manager = LoginManager(app)
login_manager.login_view = 'main.login'
csrf = CSRFProtect(app)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

from routes import main
app.register_blueprint(main)

def ensure_claim_columns():
    """Ensure all required columns exist in the claim table."""
    with app.app_context():
        if not os.path.exists(db_path):
            print("Creating new database...")
            db.create_all()

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("PRAGMA table_info(claim)")
            columns = [column[1] for column in cursor.fetchall()]

            new_columns = [
                ('vehicle_type', 'TEXT'),
                ('damage_location', 'TEXT'),
                ('damage_severity', 'TEXT'),
                ('fraud_analysis', 'TEXT'),
                ('estimated_amount', 'FLOAT DEFAULT 0.0'),
                ('damage_summary', 'TEXT'),
                ('repair_recommendation', 'TEXT'),
                ('parts_to_replace', 'TEXT')
            ]

            for column_name, column_type in new_columns:
                if column_name not in columns:
                    print(f"Adding column: {column_name}")
                    cursor.execute(f"ALTER TABLE claim ADD COLUMN {column_name} {column_type}")
                else:
                    print(f"Column {column_name} already exists")

            conn.commit()
            print("Database migration completed!")

        except Exception as e:
            print(f"Migration error: {e}")
            conn.rollback()

        finally:
            conn.close()

with app.app_context():
    db.create_all()
    ensure_claim_columns()

@app.route('/health')
def health_check():
    return 'Server is running', 200

if __name__ == '__main__':
    print('Starting Flask server on http://127.0.0.1:5000')
    app.run(host='127.0.0.1', port=5000, debug=True)