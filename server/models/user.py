import json
from datetime import datetime

from models import db
from werkzeug.security import check_password_hash, generate_password_hash


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.String(36), primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    name = db.Column(db.String(100), nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    preferences = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.now())
    updated_at = db.Column(
        db.DateTime, default=datetime.now(), onupdate=datetime.now()
    )
    is_active = db.Column(db.Boolean, default=True)

    chat_sessions = db.relationship(
        "ChatSession", backref="user", lazy=True, cascade="all, delete-orphan"
    )

    def __init__(self, id, email, name, password=None):
        self.id = id
        self.email = email
        self.name = name
        if password:
            self.set_password(password)
        self.preferences = json.dumps(
            {"favoriteCategories": [], "priceRange": [0, 2000], "favoriteBrands": []}
        )

    def set_password(self, password):
        """Hash and set password"""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        """Check if provided password matches hash"""
        return check_password_hash(self.password_hash, password)

    def get_preferences(self):
        """Get user preferences as dict"""
        try:
            return json.loads(self.preferences) if self.preferences else {}
        except json.JSONDecodeError:
            return {}

    def set_preferences(self, preferences_dict):
        """Set user preferences from dict"""
        self.preferences = json.dumps(preferences_dict)
        self.updated_at = datetime.utcnow()

    def to_dict(self):
        """Convert user to dictionary"""
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name,
            "preferences": self.get_preferences(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "is_active": self.is_active,
        }

    def __repr__(self):
        return f"<User {self.email}>"
