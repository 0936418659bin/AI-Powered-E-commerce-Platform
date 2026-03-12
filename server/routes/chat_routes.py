from flask import Blueprint, request, jsonify
import traceback
from flask_jwt_extended import jwt_required, get_jwt_identity, verify_jwt_in_request
import logging
import uuid

from services.chat_service import ChatService
from models.chat_session import ChatSession

logger = logging.getLogger(__name__)
chat_bp = Blueprint("chat", __name__)
chat_service = ChatService()


@chat_bp.route("/message", methods=["POST"])
def send_message():
    """Send a message to the chatbot"""
    try:
        # log raw body for debugging malformed JSON / client issues
        try:
            raw_body = request.data.decode("utf-8")
        except Exception:
            raw_body = str(request.data)
        logger.info(f"Raw request body: {raw_body}")

        data = request.get_json()

        if not data or not data.get("message"):
            return jsonify(
                {"success": False, "message": "Message content is required"}
            ), 400

        user_message = data["message"]
        session_id = data.get("session_id", str(uuid.uuid4()))

        user_id = None
        try:
            verify_jwt_in_request(optional=True)
            user_id = get_jwt_identity()
        except:
            pass

        response = chat_service.process_message(session_id, user_message, user_id)

        return jsonify(
            {"success": True, "response": response, "session_id": session_id}
        ), 200

    except Exception as e:
        logger.exception("Error in send_message endpoint")
        return jsonify({"success": False, "message": "Failed to process message"}), 500


@chat_bp.route("/history/<session_id>", methods=["GET"])
def get_chat_history(session_id):
    """Get chat history for a session"""
    try:
        limit = request.args.get("limit", 50, type=int)

        user_id = None
        try:
            verify_jwt_in_request(optional=True)
            user_id = get_jwt_identity()
        except:
            pass

        chat_session = ChatSession.query.get(session_id)
        if not chat_session:
            return jsonify({"success": False, "message": "Chat session not found"}), 404

        if chat_session.user_id and chat_session.user_id != user_id:
            return jsonify({"success": False, "message": "Access denied"}), 403

        history = chat_service.get_chat_history(session_id, limit)

        return jsonify(
            {"success": True, "history": history, "session": chat_session.to_dict()}
        ), 200

    except Exception as e:
        logger.error(f"Error in get_chat_history endpoint: {str(e)}")
        return jsonify({"success": False, "message": "Failed to get chat history"}), 500


@chat_bp.route("/sessions", methods=["GET"])
@jwt_required()
def get_user_sessions():
    """Get all chat sessions for the authenticated user"""
    try:
        current_user_id = get_jwt_identity()

        sessions = (
            ChatSession.query.filter_by(user_id=current_user_id)
            .order_by(ChatSession.updated_at.desc())
            .all()
        )

        return jsonify(
            {"success": True, "sessions": [session.to_dict() for session in sessions]}
        ), 200

    except Exception as e:
        logger.error(f"Error in get_user_sessions endpoint: {str(e)}")
        return jsonify(
            {"success": False, "message": "Failed to get user sessions"}
        ), 500


@chat_bp.route("/sessions/<session_id>", methods=["DELETE"])
def delete_session(session_id):
    """Delete a chat session"""
    try:
        user_id = None
        try:
            verify_jwt_in_request(optional=True)
            user_id = get_jwt_identity()
        except:
            pass

        chat_session = ChatSession.query.get(session_id)
        if not chat_session:
            return jsonify({"success": False, "message": "Chat session not found"}), 404

        if chat_session.user_id and chat_session.user_id != user_id:
            return jsonify({"success": False, "message": "Access denied"}), 403

        chat_service.clear_session_memory(session_id)

        from app import db

        db.session.delete(chat_session)
        db.session.commit()

        return jsonify(
            {"success": True, "message": "Chat session deleted successfully"}
        ), 200

    except Exception as e:
        logger.error(f"Error in delete_session endpoint: {str(e)}")
        return jsonify({"success": False, "message": "Failed to delete session"}), 500


@chat_bp.route("/sessions/<session_id>/clear", methods=["POST"])
def clear_session(session_id):
    """Clear chat history for a session"""
    try:
        user_id = None
        try:
            verify_jwt_in_request(optional=True)
            user_id = get_jwt_identity()
        except:
            pass

        chat_session = ChatSession.query.get(session_id)
        if not chat_session:
            return jsonify({"success": False, "message": "Chat session not found"}), 404

        if chat_session.user_id and chat_session.user_id != user_id:
            return jsonify({"success": False, "message": "Access denied"}), 403

        from models.message import Message
        from app import db

        Message.query.filter_by(chat_session_id=session_id).delete()
        chat_service.clear_session_memory(session_id)

        db.session.commit()

        return jsonify(
            {"success": True, "message": "Chat session cleared successfully"}
        ), 200

    except Exception as e:
        logger.error(f"Error in clear_session endpoint: {str(e)}")
        return jsonify({"success": False, "message": "Failed to clear session"}), 500


@chat_bp.route("/health", methods=["GET"])
def chat_health():
    """Check chat service health"""
    try:
        if not chat_service.initialized:
            chat_service.initialize()

        vector_stats = chat_service.vector_service.get_index_stats()

        return jsonify(
            {
                "success": True,
                "status": "healthy",
                "services": {
                    "chat_service": "initialized"
                    if chat_service.initialized
                    else "not_initialized",
                    "vector_service": "initialized"
                    if chat_service.vector_service.initialized
                    else "not_initialized",
                    "llm": "connected" if chat_service.llm else "not_connected",
                },
                "vector_stats": vector_stats,
            }
        ), 200

    except Exception as e:
        logger.error(f"Error in chat_health endpoint: {str(e)}")
        return jsonify({"success": False, "status": "unhealthy", "error": str(e)}), 500
