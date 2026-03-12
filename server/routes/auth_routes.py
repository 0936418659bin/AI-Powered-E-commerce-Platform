from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
import logging

from services.auth_service import AuthService

logger = logging.getLogger(__name__)
auth_bp = Blueprint("auth", __name__)


@auth_bp.route("/register", methods=["POST"])
def register():
    """Register a new user"""
    try:
        data = request.get_json()

        required_fields = ["name", "email", "password"]
        if not all(field in data for field in required_fields):
            return jsonify(
                {"success": False, "message": "Missing required fields"}
            ), 400

        if len(data["password"]) < 6:
            return jsonify(
                {
                    "success": False,
                    "message": "Password must be at least 6 characters long",
                }
            ), 400

        result = AuthService.register_user(
            name=data["name"], email=data["email"], password=data["password"]
        )

        if result["success"]:
            return jsonify(result), 201
        else:
            return jsonify(result), 400

    except Exception as e:
        logger.error(f"Error in register endpoint: {str(e)}")
        return jsonify({"success": False, "message": "Registration failed"}), 500


@auth_bp.route("/login", methods=["POST"])
def login():
    """Authenticate user and return tokens"""
    try:
        data = request.get_json()

        if not data.get("email") or not data.get("password"):
            return jsonify(
                {"success": False, "message": "Email and password are required"}
            ), 400

        result = AuthService.login_user(email=data["email"], password=data["password"])

        if result["success"]:
            return jsonify(result), 200
        else:
            return jsonify(result), 401

    except Exception as e:
        logger.error(f"Error in login endpoint: {str(e)}")
        return jsonify({"success": False, "message": "Login failed"}), 500


@auth_bp.route("/refresh", methods=["POST"])
@jwt_required(refresh=True)
def refresh():
    """Refresh access token"""
    try:
        current_user_id = get_jwt_identity()
        result = AuthService.refresh_token(current_user_id)

        if result["success"]:
            return jsonify(result), 200
        else:
            return jsonify(result), 401

    except Exception as e:
        logger.error(f"Error in refresh endpoint: {str(e)}")
        return jsonify({"success": False, "message": "Token refresh failed"}), 500


@auth_bp.route("/me", methods=["GET"])
@jwt_required()
def get_current_user():
    """Get current user information"""
    try:
        current_user_id = get_jwt_identity()
        user = AuthService.get_user_by_id(current_user_id)

        if user:
            return jsonify({"success": True, "user": user.to_dict()}), 200
        else:
            return jsonify({"success": False, "message": "User not found"}), 404

    except Exception as e:
        logger.error(f"Error in get_current_user endpoint: {str(e)}")
        return jsonify(
            {"success": False, "message": "Failed to get user information"}
        ), 500


@auth_bp.route("/preferences", methods=["PUT"])
@jwt_required()
def update_preferences():
    """Update user preferences"""
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json()

        if not data:
            return jsonify(
                {"success": False, "message": "Preferences data is required"}
            ), 400

        result = AuthService.update_user_preferences(current_user_id, data)

        if result["success"]:
            return jsonify(result), 200
        else:
            return jsonify(result), 400

    except Exception as e:
        logger.error(f"Error in update_preferences endpoint: {str(e)}")
        return jsonify(
            {"success": False, "message": "Failed to update preferences"}
        ), 500


@auth_bp.route("/deactivate", methods=["POST"])
@jwt_required()
def deactivate_account():
    """Deactivate user account"""
    try:
        current_user_id = get_jwt_identity()
        result = AuthService.deactivate_user(current_user_id)

        if result["success"]:
            return jsonify(result), 200
        else:
            return jsonify(result), 400

    except Exception as e:
        logger.error(f"Error in deactivate_account endpoint: {str(e)}")
        return jsonify(
            {"success": False, "message": "Failed to deactivate account"}
        ), 500
