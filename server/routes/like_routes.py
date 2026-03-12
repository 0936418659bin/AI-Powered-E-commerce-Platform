from flask import Blueprint, jsonify, request
from flask_jwt_extended import get_jwt_identity, jwt_required
from services.like_service import LikeService

like_bp = Blueprint("likes", __name__)
like_service = LikeService()


@like_bp.route("/likes/toggle", methods=["POST"])
@jwt_required()
def toggle_like():
    try:
        current_user = get_jwt_identity()
        data = request.get_json()
        product_id = data.get("product_id")
        
        if not product_id:
            return jsonify({"error": "product_id is required"}), 400
        
        is_liked, message = like_service.toggle_like(current_user, product_id)
        
        return jsonify({
            "success": True,
            "is_liked": is_liked,
            "message": message
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@like_bp.route("/likes/user/<user_id>", methods=["GET"])
@jwt_required()
def get_user_likes(user_id):
    try:
        current_user = get_jwt_identity()
        
        # Users can only access their own likes
        if current_user != user_id:
            return jsonify({"error": "Unauthorized"}), 403
        
        likes = like_service.get_user_likes(user_id)
        
        return jsonify({
            "success": True,
            "likes": likes
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@like_bp.route("/likes/product/<product_id>", methods=["GET"])
def get_product_likes(product_id):
    try:
        likes_count = like_service.get_product_likes_count(product_id)
        
        return jsonify({
            "success": True,
            "product_id": product_id,
            "likes_count": likes_count
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@like_bp.route("/likes/check", methods=["POST"])
@jwt_required()
def check_like_status():
    try:
        current_user = get_jwt_identity()
        data = request.get_json()
        product_id = data.get("product_id")
        
        if not product_id:
            return jsonify({"error": "product_id is required"}), 400
        
        is_liked = like_service.is_liked_by_user(current_user, product_id)
        
        return jsonify({
            "success": True,
            "is_liked": is_liked
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@like_bp.route("/likes/popular", methods=["GET"])
def get_popular_products():
    try:
        limit = request.args.get("limit", 10, type=int)
        popular_products = like_service.get_popular_products(limit)
        
        return jsonify({
            "success": True,
            "products": popular_products
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
