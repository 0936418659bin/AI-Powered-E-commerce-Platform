import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from flask import current_app
# Import AgentType from possible locations for compatibility across langchain versions
try:
    from langchain.agents import AgentType, initialize_agent
except Exception:
    try:
        # newer/langchain layout
        from langchain.agents.agent_types import AgentType
        from langchain.agents import initialize_agent
    except Exception:
        # Fallback: initialize_agent may still be available
        try:
            from langchain.agents import initialize_agent
            AgentType = None
        except Exception:
            initialize_agent = None
            AgentType = None
try:
    from langchain.memory import ConversationBufferWindowMemory
except Exception:
    try:
        from langchain.memory.chat_memory import ConversationBufferWindowMemory
    except Exception:
        # Minimal fallback implementation used when langchain memory is unavailable.
        class ConversationBufferWindowMemory:
            def __init__(self, k=10, return_messages=True, memory_key="chat_history"):
                self.k = k
                self.return_messages = return_messages
                self.memory_key = memory_key
                self.buffer = []

            def load_memory_variables(self, inputs):
                return {self.memory_key: self.buffer}

            def save_context(self, inputs, outputs):
                # store as simple strings for compatibility
                user = inputs.get("input") if isinstance(inputs, dict) else str(inputs)
                bot = outputs if isinstance(outputs, str) else str(outputs)
                if user:
                    self.buffer.append(user)
                if bot:
                    self.buffer.append(bot)
try:
    from langchain.schema import AIMessage, HumanMessage, SystemMessage
except Exception:
    AIMessage = None
    HumanMessage = None
    SystemMessage = None

try:
    from langchain.tools import Tool
except Exception:
    # Minimal fallback Tool class
    class Tool:
        def __init__(self, name: str, description: str, func):
            self.name = name
            self.description = description
            self.func = func

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:
    ChatGoogleGenerativeAI = None
from models.chat_session import ChatSession
from models.message import Message
from models.product import Product

from .cart_service import CartService
from .product_service import ProductService
from .vector_service import VectorService

logger = logging.getLogger(__name__)


class ChatService:
    """Enhanced chat service with LangChain and Gemini integration"""

    def __init__(self):
        self.llm = None
        self.vector_service = VectorService()
        self.product_service = ProductService()
        self.cart_service = CartService()
        self.memory_sessions = {}
        self.initialized = False

    def initialize(self):
        """Initialize LangChain components"""
        try:

            if ChatGoogleGenerativeAI is None or not current_app.config.get("GOOGLE_API_KEY"):
                logger.warning("ChatGoogleGenerativeAI or GOOGLE_API_KEY not available; LLM disabled")
                self.llm = None
            else:
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash",
                    google_api_key=current_app.config["GOOGLE_API_KEY"],
                    temperature=0.7,
                    max_tokens=1000,
                    convert_system_message_to_human=True,
                )

            # initialize vector service (embedding model) - may work without Pinecone
            try:
                self.vector_service.initialize()
            except Exception as e:
                logger.warning(f"Vector service failed to initialize: {e}")

            self.initialized = True
            logger.info("Chat service initialized (LLM %s)", "enabled" if self.llm else "disabled")

        except Exception as e:
            logger.error(f"Failed to initialize chat service: {str(e)}")
            raise

    def get_or_create_memory(self, session_id: str) -> ConversationBufferWindowMemory:
        """Get or create memory for a chat session"""
        if session_id not in self.memory_sessions:
            self.memory_sessions[session_id] = ConversationBufferWindowMemory(
                k=10,
                return_messages=True,
                memory_key="chat_history",
            )
        return self.memory_sessions[session_id]

    def create_tools(self) -> List[Tool]:
        """Create tools for the LangChain agent"""
        tools = [
            Tool(
                name="search_products",
                description="Find products using semantic search. Input: search query (str).",
                func=self._search_products_tool,
            ),
            Tool(
                name="filter_products",
                description="Filter products. Input: JSON string with keys: category, subcategory, brand, min_price, max_price, min_rating, in_stock_only, features (list), search_query, limit.",
                func=self._filter_products_tool,
            ),
            Tool(
                name="get_product_details",
                description="Get product details. Input: product ID (str).",
                func=self._get_product_details_tool,
            ),
            Tool(
                name="get_recommendations",
                description="Get recommendations. Input: product ID (str) or preference description (str).",
                func=self._get_recommendations_tool,
            ),
            Tool(
                name="add_to_cart",
                description="Add a product to the user's cart. Input: JSON string with keys: product_id (str), quantity (int, optional, default 1).",
                func=self._add_to_cart_tool,
            ),
        ]
        return tools

    def _search_products_tool(self, query: str) -> str:
        """Tool function for semantic product search"""
        try:
            similar_products = self.vector_service.search_similar_products(
                query, top_k=6
            )

            if not similar_products:
                return json.dumps(
                    {
                        "message": "No products found for the given query.",
                        "product_ids": [],
                    }
                )

            product_ids = [p["id"] for p in similar_products]
            products = Product.query.filter(Product.id.in_(product_ids)).all()

            result = "Found the following products:\n"
            for product in products:
                result += f"- {product.name} by {product.brand} - ${product.price}\n"
                result += f"  {product.description[:100]}...\n"

            return json.dumps({"message": result, "product_ids": product_ids})

        except Exception as e:
            logger.error(f"Error in search_products_tool: {str(e)}")
            return json.dumps(
                {
                    "message": "Error occurred while searching for products.",
                    "product_ids": [],
                }
            )

    def _filter_products_tool(self, filter_json: str) -> str:
        """Tool function for filtering products"""
        try:
            filters = json.loads(filter_json)
            products = Product.search_by_filters(**filters)

            if not products:
                return json.dumps(
                    {
                        "message": "No products found matching the specified filters.",
                        "product_ids": [],
                    }
                )

            result = f"Found {len(products)} products matching your criteria:\n"
            for product in products[:5]:
                result += f"- {product.name} by {product.brand} - ${product.price}\n"

            product_ids = [product.id for product in products[:5]]
            return json.dumps({"message": result, "product_ids": product_ids})

        except Exception as e:
            logger.error(f"Error in filter_products_tool: {str(e)}")
            return json.dumps(
                {
                    "message": "Error occurred while filtering products.",
                    "product_ids": [],
                }
            )

    def _get_product_details_tool(self, product_id: str) -> str:
        """Tool function for getting product details"""
        try:
            product = Product.query.get(product_id.strip())
            if not product:
                return "Product not found."

            result = "Product Details:\n"
            result += f"Name: {product.name}\n"
            result += f"Brand: {product.brand}\n"
            result += f"Price: ${product.price}\n"
            result += f"Rating: {product.rating}/5 ({product.review_count} reviews)\n"
            result += f"Description: {product.description}\n"
            result += f"Features: {', '.join(product.get_features())}\n"
            result += f"Stock: {product.stock} available\n"

            return result

        except Exception as e:
            logger.error(f"Error in get_product_details_tool: {str(e)}")
            return "Error occurred while getting product details."

    def _get_recommendations_tool(self, input_text: str) -> str:
        """Tool function for getting product recommendations"""
        try:
            product = Product.query.get(input_text.strip())

            if product:
                similar_products = self.vector_service.search_similar_products(
                    product.get_search_text(), top_k=4
                )
                similar_ids = [
                    p["id"] for p in similar_products if p["id"] != product.id
                ]
                recommendations = Product.query.filter(
                    Product.id.in_(similar_ids)
                ).all()
            else:
                similar_products = self.vector_service.search_similar_products(
                    input_text, top_k=4
                )
                similar_ids = [p["id"] for p in similar_products]
                recommendations = Product.query.filter(
                    Product.id.in_(similar_ids)
                ).all()

            if not recommendations:
                return "No recommendations found."

            result = "Here are some recommendations:\n"
            for rec in recommendations:
                result += f"- {rec.name} by {rec.brand} - ${rec.price}\n"

            return result

        except Exception as e:
            logger.error(f"Error in get_recommendations_tool: {str(e)}")
            return "Error occurred while getting recommendations."

    def _add_to_cart_tool(self, input_json: str) -> str:
        """Tool function to add a product to the user's cart"""
        try:
            # Log the input for debugging
            logger.info(f"add_to_cart_tool input: {input_json}")
            
            data = json.loads(input_json)
            product_id = data.get("product_id")
            quantity = data.get("quantity", 1)
            user_id = data.get("user_id", "guest_user")

            logger.info(f"Parsed data: product_id={product_id}, quantity={quantity}, user_id={user_id}")

            if not product_id:
                return json.dumps(
                    {"message": "Missing product_id for add to cart.", "success": False}
                )

            # If product_id looks like a product name, try to find the actual product
            if len(product_id) < 32 or " " in product_id:
                logger.info(f"Searching for product by name: {product_id}")
                # Search for product by name (case-insensitive)
                product = Product.query.filter(
                    Product.name.ilike(f"%{product_id}%")
                ).first()
                
                if product:
                    logger.info(f"Found product: {product.name} with ID: {product.id}")
                    product_id = product.id
                else:
                    logger.warning(f"Product not found: {product_id}")
                    return json.dumps(
                        {
                            "message": f"Product '{product_id}' not found.",
                            "success": False,
                        }
                    )

            # Add to cart using the cart service
            logger.info(f"Adding to cart: user_id={user_id}, product_id={product_id}, quantity={quantity}")
            result = self.cart_service.add_to_cart(user_id, product_id, quantity)
            logger.info(f"Cart service result: {result}")
            
            # Check if the cart service returned an error
            if not result.get("success", True):
                return json.dumps(result)
            
            # Get product details for response
            product = Product.query.get(product_id)
            if not product:
                return json.dumps(
                    {
                        "message": f"Product with ID {product_id} not found.",
                        "success": False,
                    }
                )

            success_response = {
                "message": f"Added {quantity} x {product.name} to your cart.",
                "success": True,
                "product": {
                    "id": product.id,
                    "name": product.name,
                    "price": product.price,
                },
                "quantity": quantity,
            }
            
            logger.info(f"Returning success response: {success_response}")
            return json.dumps(success_response)
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in add_to_cart_tool: {str(e)}")
            logger.error(f"Input that caused error: {repr(input_json)}")
            return json.dumps(
                {"message": "Invalid JSON format in request.", "success": False}
            )
        except Exception as e:
            logger.error(f"Error in add_to_cart_tool: {str(e)}")
            return json.dumps(
                {"message": "Error occurred while adding to cart.", "success": False}
            )

    def _extract_product_names_from_text(self, text: str) -> list:
        """Extract product names from the message text by matching against all product names in the database."""
        product_names = []
        all_products = Product.query.all()
        for product in all_products:
            if product.name in text:
                product_names.append(product.name)
        return product_names

    def process_message(
        self, session_id: str, user_message: str, user_id: str = None
    ) -> Dict[str, Any]:
        """Process user message and generate AI response"""
        if not self.initialized:
            self.initialize()

        try:
            chat_session = ChatSession.query.get(session_id)
            if not chat_session:
                chat_session = ChatSession(id=session_id, user_id=user_id)
                from app import db

                db.session.add(chat_session)
                db.session.commit()

            user_msg = Message(
                id=str(uuid.uuid4()),
                chat_session_id=session_id,
                content=user_message,
                is_bot=False,
            )
            from app import db

            db.session.add(user_msg)

            memory = self.get_or_create_memory(session_id)
            chat_history = []
            if hasattr(memory, "buffer"):
                for msg in memory.buffer:
                    if hasattr(msg, "content"):
                        chat_history.append(msg.content)
                    elif isinstance(msg, str):
                        chat_history.append(msg)

            tools = self.create_tools()

            # If initialize_agent or LLM is not available, use a simple fallback handler
            if initialize_agent is None or self.llm is None:
                logger.info("Using fallback chat handler (no langchain agent available)")

                # Try semantic search using vector service
                try:
                    similar = self.vector_service.search_similar_products(user_message, top_k=6)
                except Exception:
                    similar = []

                if similar:
                    product_ids = [p["id"] for p in similar]
                    products = Product.query.filter(Product.id.in_(product_ids)).all()
                    message_text = "Tôi tìm thấy các sản phẩm sau phù hợp với yêu cầu của bạn:\n"
                    for product in products[:6]:
                        message_text += f"- {product.name} by {product.brand} - ${product.price}\n"
                    result = {"output": message_text}
                else:
                    # No semantic results — fallback generic reply
                    message_text = "Xin lỗi, chức năng trợ lý nâng cao chưa được cấu hình. Bạn có thể tìm sản phẩm qua API /api/products hoặc mô tả rõ hơn yêu cầu."
                    result = {"output": message_text}
            ai_response = (
                result["output"] if isinstance(result, dict) and "output" in result else result
            )

            product_ids = []
            if isinstance(result, dict) and "intermediate_steps" in result:
                for step in result["intermediate_steps"]:
                    tool_name = (
                        getattr(step[0], "tool", None)
                        if hasattr(step[0], "tool")
                        else None
                    )
                    tool_output = step[1]
                    if tool_name in ["search_products", "filter_products"]:
                        try:
                            parsed = json.loads(tool_output)
                            ids = parsed.get("product_ids", [])
                            if ids:
                                product_ids.extend(ids)
                        except Exception:
                            pass
            product_ids = list(dict.fromkeys(product_ids))

            message_text = ai_response
            if not product_ids:
                try:
                    parsed = json.loads(ai_response)
                    message_text = parsed.get("message", ai_response)
                    product_ids = parsed.get("product_ids", [])
                except Exception:
                    pass

            if not product_ids:
                product_names = self._extract_product_names_from_text(message_text)
                if product_names:
                    product_ids = [
                        p.id
                        for p in Product.query.filter(
                            Product.name.in_(product_names)
                        ).all()
                    ]

            ai_msg = Message(
                id=str(uuid.uuid4()),
                chat_session_id=session_id,
                content=message_text,
                is_bot=True,
                message_type="product" if product_ids else "text",
                products=product_ids,
            )
            db.session.add(ai_msg)
            db.session.commit()

            products = []
            if product_ids:
                products = [
                    Product.query.get(pid).to_dict()
                    for pid in product_ids
                    if Product.query.get(pid)
                ]

            return {
                "id": ai_msg.id,
                "content": message_text,
                "isBot": True,
                "timestamp": ai_msg.created_at.isoformat(),
                "products": products,
                "type": ai_msg.message_type,
            }

        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            error_msg = Message(
                id=str(uuid.uuid4()),
                chat_session_id=session_id,
                content="I'm sorry, I encountered an error. Please try again.",
                is_bot=True,
            )
            from app import db

            db.session.add(error_msg)
            db.session.commit()
            return {
                "id": error_msg.id,
                "content": error_msg.content,
                "isBot": True,
                "timestamp": error_msg.created_at.isoformat(),
                "products": [],
                "type": "text",
            }

    def _extract_product_ids_from_response(self, response: str) -> List[str]:
        """Extract product IDs from AI response (basic implementation)"""

        product_ids = []

        return product_ids

    def get_chat_history(
        self, session_id: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get chat history for a session"""
        try:
            messages = (
                Message.query.filter_by(chat_session_id=session_id)
                .order_by(Message.created_at.asc())
                .limit(limit)
                .all()
            )

            return [msg.to_dict(include_product_details=True) for msg in messages]

        except Exception as e:
            logger.error(f"Error getting chat history: {str(e)}")
            return []

    def clear_session_memory(self, session_id: str):
        """Clear memory for a specific session"""
        if session_id in self.memory_sessions:
            del self.memory_sessions[session_id]
