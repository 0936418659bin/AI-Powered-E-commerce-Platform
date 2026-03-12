from datetime import datetime

from models import db


class UserLike(db.Model):
    __tablename__ = "user_likes"

    id = db.Column(db.String(36), primary_key=True)
    user_id = db.Column(db.String(36), db.ForeignKey("users.id"), nullable=False)
    product_id = db.Column(db.String(36), db.ForeignKey("products.id"), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now())
    
    # Ensure a user can only like a product once
    __table_args__ = (db.UniqueConstraint('user_id', 'product_id', name='unique_user_product_like'),)

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "product_id": self.product_id,
            "created_at": self.created_at.isoformat(),
        }
