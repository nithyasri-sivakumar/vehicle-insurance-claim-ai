from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

db = SQLAlchemy()

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)

    def __repr__(self):
        return f'<User {self.username}>'

class Claim(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    description = db.Column(db.Text, nullable=False)
    image_path = db.Column(db.String(300), nullable=False)
    status = db.Column(db.String(50), default='pending')  # pending, approved, no_damage, needs_review, fraud
    severity = db.Column(db.String(50))  # None, Minor, Moderate, Severe, Total Loss
    mismatch = db.Column(db.Boolean, default=False)
    fraud_score = db.Column(db.Float, default=0.0)
    estimated_amount = db.Column(db.Float, default=0.0)
    damage_summary = db.Column(db.Text)
    repair_recommendation = db.Column(db.Text)
    parts_to_replace = db.Column(db.Text)
    vehicle_type = db.Column(db.String(50))  # car, bike, truck, etc.
    damage_location = db.Column(db.String(200))  # front bumper, hood, door, headlight, side panel
    damage_severity = db.Column(db.String(50))  # None, Minor, Moderate, Severe, Total Loss
    fraud_analysis = db.Column(db.Text)  # JSON string of fraud analysis details
    submitted_at = db.Column(db.DateTime, default=db.func.current_timestamp())

    user = db.relationship('User', backref=db.backref('claims', lazy=True))

    def __repr__(self):
        return f'<Claim {self.id}>'
