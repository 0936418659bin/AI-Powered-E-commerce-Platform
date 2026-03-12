import uuid

from app import db
from models import Product, UserLike


class LikeService:
    def toggle_like(self, user_id: str, product_id: str):
        """Toggle like status for a product"""
        existing_like = UserLike.query.filter_by(
            user_id=user_id, product_id=product_id
        ).first()
        
        if existing_like:
            # Unlike the product
            db.session.delete(existing_like)
            db.session.commit()
            return False, "Product unliked"
        else:
            # Like the product
            new_like = UserLike(
                id=str(uuid.uuid4()),
                user_id=user_id,
                product_id=product_id,
            )
            db.session.add(new_like)
            db.session.commit()
            return True, "Product liked"

    def get_user_likes(self, user_id: str):
        """Get all products liked by a user"""
        likes = UserLike.query.filter_by(user_id=user_id).all()
        
        return [
            {
                **like.to_dict(),
                "product": Product.query.get(like.product_id).to_dict()
                if Product.query.get(like.product_id)
                else None,
            }
            for like in likes
        ]

    def is_liked_by_user(self, user_id: str, product_id: str):
        """Check if a product is liked by a user"""
        like = UserLike.query.filter_by(
            user_id=user_id, product_id=product_id
        ).first()
        return like is not None

    def get_product_likes_count(self, product_id: str):
        """Get the number of likes for a product"""
        return UserLike.query.filter_by(product_id=product_id).count()

    def get_popular_products(self, limit: int = 10):
        """Get most liked products"""
        from sqlalchemy import func
        
        popular_products = (
            db.session.query(
                UserLike.product_id,
                func.count(UserLike.id).label('likes_count')
            )
            .group_by(UserLike.product_id)
            .order_by(func.count(UserLike.id).desc())
            .limit(limit)
            .all()
        )
        
        result = []
        for product_id, likes_count in popular_products:
            product = Product.query.get(product_id)
            if product:
                product_dict = product.to_dict()
                product_dict['likes_count'] = likes_count
                result.append(product_dict)
        
        return result
