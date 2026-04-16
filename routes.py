from database.models import db, User, Claim
from flask import Blueprint, render_template, redirect, url_for, flash, request, current_app
from flask_login import login_user, login_required, logout_user, current_user
import bcrypt
from werkzeug.utils import secure_filename
import os
import json
from models.claim_pipeline import analyze_claim, infer_vehicle_type_from_text
from models.vehicle_detector import is_vehicle_image
from datetime import datetime

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        existing = User.query.filter((User.username == username) | (User.email == email)).first()
        if existing:
            flash('Username or email already exists.', 'danger')
            return render_template('register.html')

        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        user = User(username=username, email=email, password=hashed_password.decode('utf-8'))
        db.session.add(user)
        db.session.commit()
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('main.login'))
    return render_template('register.html')

@main.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and bcrypt.checkpw(password.encode('utf-8'), user.password.encode('utf-8')):
            login_user(user)
            return redirect(url_for('main.dashboard'))
        flash('Invalid credentials', 'danger')
    return render_template('login.html')

@main.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('main.index'))

@main.route('/dashboard')
@login_required
def dashboard():
    claims = Claim.query.filter_by(user_id=current_user.id).all()
    return render_template('dashboard.html', claims=claims)

@main.route('/admin')
@login_required
def admin():
    if current_user.email != 'admin@example.com':  # Simple admin check
        flash('Access denied', 'danger')
        return redirect(url_for('main.dashboard'))
    claims = Claim.query.all()
    return render_template('admin.html', claims=claims)

@main.route('/submit_claim', methods=['GET', 'POST'])
@login_required
def submit_claim():
    if request.method == 'POST':
        description = request.form['description']
        file = request.files.get('image')
        if not file or file.filename == '':
            flash('Please upload a damage image before submitting.', 'danger')
            return render_template('submit_claim.html')

        filename = secure_filename(file.filename)
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        described_vehicle_type = infer_vehicle_type_from_text(description)

        # Step 2: Vehicle detection validation
        confidence_threshold = 0.28 if described_vehicle_type == 'bike' else 0.30
        is_vehicle, confidence, detected_class = is_vehicle_image(filepath, confidence_threshold=confidence_threshold)
        print(f"[Route] Vehicle detected={is_vehicle}, class={detected_class}, confidence={confidence:.2f}")

        near_miss_two_wheeler = (
            not is_vehicle
            and confidence >= 0.25  # Lower threshold for two-wheelers
            and described_vehicle_type == 'bike'
        )

        if near_miss_two_wheeler:
            is_vehicle = True
            detected_class = 'motorcycle'
            flash('Two-wheeler image matched the description with low confidence and was sent for review.', 'warning')

        # Correct misclassification for two-wheelers
        if is_vehicle and detected_class == 'car' and described_vehicle_type == 'bike' and confidence < 0.6:
            detected_class = 'motorcycle'
            print(f"[Route] Corrected detected class from 'car' to 'motorcycle' based on description")

        if not is_vehicle:
            os.remove(filepath)
            flash('Invalid image. Please upload a valid vehicle image.', 'danger')
            return render_template('submit_claim.html')

        if is_vehicle and confidence < 0.5 and not near_miss_two_wheeler:
            flash('Low confidence detection. Detected vehicle image will still be processed.', 'warning')

        claim = Claim(user_id=current_user.id, description=description, image_path=filename, status='pending')
        db.session.add(claim)
        db.session.commit()

        analysis = analyze_claim(
            filepath,
            description,
            inferred_vehicle_type=detected_class or described_vehicle_type,
        )

        claim.status = analysis.get('status', 'needs_review')
        claim.severity = analysis.get('severity', 'Unknown')
        claim.mismatch = analysis.get('mismatch', False)
        claim.fraud_score = analysis.get('fraud_score', 0.0)
        claim.estimated_amount = analysis.get('estimated_amount', 0.0)
        claim.damage_summary = analysis.get('damage_summary', 'Damage summary unavailable.')
        claim.repair_recommendation = analysis.get('repair_recommendation', 'Manual inspection required.')
        claim.parts_to_replace = analysis.get('parts_to_replace', 'None')
        claim.vehicle_type = analysis.get('vehicle_type', detected_class or 'unknown')
        claim.damage_location = analysis.get('damage_location', 'None')
        claim.damage_severity = analysis.get('damage_severity', analysis.get('severity', 'Unknown'))
        claim.fraud_analysis = json.dumps(analysis.get('fraud_analysis', {}))
        db.session.commit()
        flash('Your claim has been submitted and processed.', 'success')
        return redirect(url_for('main.dashboard'))
    return render_template('submit_claim.html')
