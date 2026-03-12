import uuid

from app import db
from models import Cart, Product


class CartService:
    def add_to_cart(self, user_id: str, product_id: str, quantity: int = 1):
        """Add a product to the user's cart"""
        try:
            # Check if product exists
            product = Product.query.get(product_id)
            if not product:
                return {"success": False, "message": "Product not found"}
            
            # Check if item already exists in cart
            cart_item = Cart.query.filter_by(user_id=user_id, product_id=product_id).first()
            
            if cart_item:
                # Update existing item
                cart_item.quantity += quantity
                cart_item.updated_at = db.func.now()
            else:
                # Create new cart item
                cart_item = Cart(
                    id=str(uuid.uuid4()),
                    user_id=user_id,
                    product_id=product_id,
                    quantity=quantity,
                )
                db.session.add(cart_item)
            
            db.session.commit()
            return {"success": True, "message": "Product added to cart", "cart_item": cart_item.to_dict()}
            
        except Exception as e:
            db.session.rollback()
            return {"success": False, "message": f"Error adding to cart: {str(e)}"}

    def get_cart(self, user_id: str):
        """Get all items in the user's cart"""
        try:
            cart_items = Cart.query.filter_by(user_id=user_id).all()
            
            cart_data = []
            for item in cart_items:
                product = Product.query.get(item.product_id)
                if product:
                    cart_data.append({
                        **item.to_dict(),
                        "product": product.to_dict()
                    })
            
            return cart_data
            
        except Exception as e:
            return []

    def remove_from_cart(self, user_id: str, item_id: str):
        """Remove an item from the user's cart"""
        try:
            cart_item = Cart.query.filter_by(id=item_id, user_id=user_id).first()
            if cart_item:
                db.session.delete(cart_item)
                db.session.commit()
                return {"success": True, "message": "Item removed from cart"}
            return {"success": False, "message": "Item not found in cart"}
        except Exception as e:
            db.session.rollback()
            return {"success": False, "message": f"Error removing item: {str(e)}"}

    def update_cart_quantity(self, user_id: str, item_id: str, quantity: int):
        """Update the quantity of a cart item"""
        try:
            if quantity <= 0:
                return self.remove_from_cart(user_id, item_id)
            
            cart_item = Cart.query.filter_by(id=item_id, user_id=user_id).first()
            if cart_item:
                cart_item.quantity = quantity
                cart_item.updated_at = db.func.now()
                db.session.commit()
                return {"success": True, "message": "Cart updated", "cart_item": cart_item.to_dict()}
            return {"success": False, "message": "Item not found in cart"}
        except Exception as e:
            db.session.rollback()
            return {"success": False, "message": f"Error updating cart: {str(e)}"}

    def clear_cart(self, user_id: str):
        """Clear all items from the user's cart"""
        try:
            Cart.query.filter_by(user_id=user_id).delete()
            db.session.commit()
            return {"success": True, "message": "Cart cleared"}
        except Exception as e:
            db.session.rollback()
            return {"success": False, "message": f"Error clearing cart: {str(e)}"}

    def get_cart_total(self, user_id: str):
        """Calculate the total price of items in the cart"""
        try:
            cart_items = Cart.query.filter_by(user_id=user_id).all()
            total = 0
            item_count = 0
            
            for item in cart_items:
                product = Product.query.get(item.product_id)
                if product:
                    total += product.price * item.quantity
                    item_count += item.quantity
            
            return {
                "success": True,
                "total": total,
                "item_count": item_count,
                "formatted_total": f"${total:.2f}"
            }
        except Exception as e:
            return {"success": False, "message": f"Error calculating total: {str(e)}"}
