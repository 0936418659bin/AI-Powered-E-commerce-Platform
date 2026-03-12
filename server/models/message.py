import json
from datetime import datetime

from models import db


class Message(db.Model):
    __tablename__ = "messages"

    id = db.Column(db.String(36), primary_key=True)
    chat_session_id = db.Column(
        db.String(36), db.ForeignKey("chat_sessions.id"), nullable=False
    )
    content = db.Column(db.Text, nullable=False)
    is_bot = db.Column(db.Boolean, default=False)
    message_type = db.Column(db.String(50), default="text")
    products = db.Column(db.Text)
    extra_data = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.now())

    def __init__(
        self,
        id,
        chat_session_id,
        content,
        is_bot=False,
        message_type="text",
        products=None,
        extra_data=None,
    ):
        self.id = id
        self.chat_session_id = chat_session_id
        self.content = content
        self.is_bot = is_bot
        self.message_type = message_type
        self.products = json.dumps(products or [])
        self.extra_data = json.dumps(extra_data or {})

    def get_products(self):
        """Get product IDs as list"""
        try:
            return json.loads(self.products) if self.products else []
        except json.JSONDecodeError:
            return []

    def set_products(self, product_ids):
        """Set product IDs from list"""
        self.products = json.dumps(product_ids)

    def get_extra_data(self):
        """Get extra data as dict"""
        try:
            return json.loads(self.extra_data) if self.extra_data else {}
        except json.JSONDecodeError:
            return {}

    def set_extra_data(self, extra_data_dict):
        """Set extra data from dict"""
        self.extra_data = json.dumps(extra_data_dict)

    def to_dict(self, include_product_details=False):
        """Convert message to dictionary"""
        # Get product IDs by default
        products_data = self.get_products()
        
        # If include_product_details is True, replace IDs with full product objects
        if include_product_details and self.get_products():
            from .product import Product

            product_details = []
            for product_id in self.get_products():
                product = Product.query.get(product_id)
                if product:
                    product_details.append(product.to_dict())
            # Replace the product IDs with full product objects
            products_data = product_details

        data = {
            "id": self.id,
            "chatSessionId": self.chat_session_id,
            "content": self.content,
            "isBot": self.is_bot,
            "type": self.message_type,
            "products": products_data,
            "extraData": self.get_extra_data(),
            "timestamp": self.created_at.isoformat(),
        }

        return data

    def __repr__(self):
        return f"<Message {self.id}>"
