# AI-Based Vehicle Insurance Claim Validation System

A polished Flask web application for vehicle insurance claim validation using simulated Vision-Language AI logic.

## Features

- User registration and login
- Claim submission with image upload
- Image and text analysis for severity and mismatch
- Fraud scoring and claim validation
- User dashboard for personal claims
- Admin dashboard for reviewing all claims
- Improved UI with responsive dashboard and form panels

## Setup Instructions

1. Open the project folder in your terminal.
2. Create and activate the Python virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
4. Run the application:
   ```powershell
   python app.py
   ```
5. Open the browser at `http://127.0.0.1:5000`

## Admin Access

- Register an account with email `admin@example.com` to access the admin dashboard.

## Project Structure

- `app.py` — main Flask app and configuration
- `routes.py` — route definitions and claim handling
- `database/models.py` — shared SQLAlchemy models and database instance
- `models/ai_model.py` — simulated AI analysis logic
- `templates/` — Jinja2 templates using a shared base layout
- `static/` — CSS and JavaScript assets
- `requirements.txt` — Python dependency list

## Database

- Uses SQLite
- `insurance.db` is created automatically on app startup

## Notes

- The AI is simulated with rule-based logic for demo purposes.
- Replace `SECRET_KEY` in `app.py` before deploying.
- The app is designed for local development and demonstration.
