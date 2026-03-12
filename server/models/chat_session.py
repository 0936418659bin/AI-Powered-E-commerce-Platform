import json
from datetime import datetime

from models import db


class ChatSession(db.Model):
    __tablename__ = "chat_sessions"

    id = db.Column(db.String(36), primary_key=True)
    user_id = db.Column(db.String(36), db.ForeignKey("users.id"), nullable=True)
    session_data = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.now())
    updated_at = db.Column(
        db.DateTime, default=datetime.now(), onupdate=datetime.now()
    )
    is_active = db.Column(db.Boolean, default=True)

    messages = db.relationship(
        "Message", backref="chat_session", lazy=True, cascade="all, delete-orphan"
    )

    def __init__(self, id, user_id=None, session_data=None):
        self.id = id
        self.user_id = user_id
        self.session_data = json.dumps(session_data or {})

    def get_session_data(self):
        """Get session data as dict"""
        try:
            return json.loads(self.session_data) if self.session_data else {}
        except json.JSONDecodeError:
            return {}

    def set_session_data(self, data_dict):
        """Set session data from dict"""
        self.session_data = json.dumps(data_dict)
        self.updated_at = datetime.utcnow()

    def get_message_count(self):
        """Get total message count in session"""
        return len(self.messages)

    def get_recent_messages(self, limit=10):
        """Get recent messages from session"""
        from models import Message

        return (
            Message.query.filter_by(chat_session_id=self.id)
            .order_by(Message.created_at.desc())
            .limit(limit)
            .all()
        )

    def to_dict(self, include_messages=False):
        """Convert chat session to dictionary"""
        data = {
            "id": self.id,
            "userId": self.user_id,
            "sessionData": self.get_session_data(),
            "messageCount": self.get_message_count(),
            "createdAt": self.created_at.isoformat(),
            "updatedAt": self.updated_at.isoformat(),
            "isActive": self.is_active,
        }

        if include_messages:
            data["messages"] = [msg.to_dict() for msg in self.messages]

        return data

    def __repr__(self):
        return f"<ChatSession {self.id}>"
