import logging
from typing import List, Dict, Any, Optional
from models.product import Product
from .vector_service import VectorService

logger = logging.getLogger(__name__)


class ProductService:
    """Service for product-related operations"""

    def __init__(self):
        self.vector_service = VectorService()

    def create_product(self, product_data: Dict[str, Any]) -> Product:
        """Create a new product and generate its embedding"""
        try:
            product = Product(**product_data)

            from app import db

            db.session.add(product)
            db.session.flush()

            search_text = product.get_search_text()
            metadata = {
                "category": product.category,
                "subcategory": product.subcategory,
                "brand": product.brand,
                "price": product.price,
                "rating": product.rating,
                "in_stock": product.is_in_stock(),
            }

            self.vector_service.upsert_product_embedding(
                product.id, search_text, metadata
            )

            product.embedding_id = product.id
            db.session.commit()

            logger.info(f"Created product: {product.name}")
            return product

        except Exception as e:
            logger.error(f"Error creating product: {str(e)}")
            from app import db

            db.session.rollback()
            raise

    def update_product(
        self, product_id: str, update_data: Dict[str, Any]
    ) -> Optional[Product]:
        """Update a product and refresh its embedding"""
        try:
            product = Product.query.get(product_id)
            if not product:
                return None

            for key, value in update_data.items():
                if hasattr(product, key):
                    setattr(product, key, value)

            content_fields = [
                "name",
                "description",
                "features",
                "category",
                "subcategory",
                "brand",
            ]
            if any(field in update_data for field in content_fields):
                search_text = product.get_search_text()
                metadata = {
                    "category": product.category,
                    "subcategory": product.subcategory,
                    "brand": product.brand,
                    "price": product.price,
                    "rating": product.rating,
                    "in_stock": product.is_in_stock(),
                }

                self.vector_service.upsert_product_embedding(
                    product.id, search_text, metadata
                )

            from app import db

            db.session.commit()

            logger.info(f"Updated product: {product.name}")
            return product

        except Exception as e:
            logger.error(f"Error updating product: {str(e)}")
            from app import db

            db.session.rollback()
            raise

    def delete_product(self, product_id: str) -> bool:
        """Delete a product and its embedding"""
        try:
            product = Product.query.get(product_id)
            if not product:
                return False

            self.vector_service.delete_product_embedding(product_id)

            from app import db

            db.session.delete(product)
            db.session.commit()

            logger.info(f"Deleted product: {product.name}")
            return True

        except Exception as e:
            logger.error(f"Error deleting product: {str(e)}")
            from app import db

            db.session.rollback()
            raise

    def search_products(
        self, query: str, filters: Dict[str, Any] = None, limit: int = 20
    ) -> List[Product]:
        """Search products using both vector similarity and traditional filters"""
        try:
            vector_results = self.vector_service.search_similar_products(
                query,
                top_k=limit * 2,
            )

            if not vector_results:
                return Product.search_by_filters(search_query=query, limit=limit)

            product_ids = [result["id"] for result in vector_results]

            query_builder = Product.query.filter(Product.id.in_(product_ids))

            if filters:
                if filters.get("category"):
                    query_builder = query_builder.filter(
                        Product.category == filters["category"]
                    )

                if filters.get("subcategory"):
                    query_builder = query_builder.filter(
                        Product.subcategory == filters["subcategory"]
                    )

                if filters.get("brand"):
                    query_builder = query_builder.filter(
                        Product.brand == filters["brand"]
                    )

                if filters.get("min_price") is not None:
                    query_builder = query_builder.filter(
                        Product.price >= filters["min_price"]
                    )

                if filters.get("max_price") is not None:
                    query_builder = query_builder.filter(
                        Product.price <= filters["max_price"]
                    )

                if filters.get("min_rating") is not None:
                    query_builder = query_builder.filter(
                        Product.rating >= filters["min_rating"]
                    )

                if filters.get("in_stock_only"):
                    query_builder = query_builder.filter(Product.stock > 0)

            products = query_builder.all()

            product_score_map = {
                result["id"]: result["score"] for result in vector_results
            }
            products.sort(key=lambda p: product_score_map.get(p.id, 0), reverse=True)

            return products[:limit]

        except Exception as e:
            logger.error(f"Error searching products: {str(e)}")
            return []

    def get_recommendations(
        self,
        product_id: str = None,
        user_preferences: Dict[str, Any] = None,
        limit: int = 6,
    ) -> List[Product]:
        """Get product recommendations"""
        try:
            if product_id:
                product = Product.query.get(product_id)
                if not product:
                    return []

                search_text = product.get_search_text()
                similar_results = self.vector_service.search_similar_products(
                    search_text,
                    top_k=limit + 1,
                )

                similar_ids = [
                    r["id"] for r in similar_results if r["id"] != product_id
                ]

            elif user_preferences:
                pref_text = self._build_preference_text(user_preferences)
                similar_results = self.vector_service.search_similar_products(
                    pref_text, top_k=limit
                )
                similar_ids = [r["id"] for r in similar_results]

            else:
                return (
                    Product.query.filter(Product.is_active == True)
                    .order_by(Product.rating.desc())
                    .limit(limit)
                    .all()
                )

            products = Product.query.filter(Product.id.in_(similar_ids)).all()

            if similar_results:
                score_map = {r["id"]: r["score"] for r in similar_results}
                products.sort(key=lambda p: score_map.get(p.id, 0), reverse=True)

            return products[:limit]

        except Exception as e:
            logger.error(f"Error getting recommendations: {str(e)}")
            return []

    def _build_preference_text(self, preferences: Dict[str, Any]) -> str:
        """Build search text from user preferences"""
        text_parts = []

        if preferences.get("favoriteCategories"):
            text_parts.extend(preferences["favoriteCategories"])

        if preferences.get("favoriteBrands"):
            text_parts.extend(preferences["favoriteBrands"])

        if preferences.get("priceRange"):
            min_price, max_price = preferences["priceRange"]
            if max_price < 500:
                text_parts.append("budget affordable cheap")
            elif max_price > 1500:
                text_parts.append("premium high-end expensive")
            else:
                text_parts.append("mid-range")

        return " ".join(text_parts) if text_parts else "popular electronics"

    def bulk_generate_embeddings(self):
        """Generate embeddings for all products (useful for initial setup)"""
        try:
            products = Product.query.filter(Product.is_active == True).all()

            batch_data = []
            for product in products:
                search_text = product.get_search_text()
                metadata = {
                    "category": product.category,
                    "subcategory": product.subcategory,
                    "brand": product.brand,
                    "price": product.price,
                    "rating": product.rating,
                    "in_stock": product.is_in_stock(),
                }

                batch_data.append(
                    {"id": product.id, "text": search_text, "metadata": metadata}
                )

            self.vector_service.batch_upsert_products(batch_data)

            for product in products:
                product.embedding_id = product.id

            from app import db

            db.session.commit()

            logger.info(f"Generated embeddings for {len(products)} products")
            return len(products)

        except Exception as e:
            logger.error(f"Error generating bulk embeddings: {str(e)}")
            raise
