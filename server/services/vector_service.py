import logging
from typing import Any, Dict, List

from flask import current_app
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

logger = logging.getLogger(__name__)


class VectorService:
    """Service for managing vector embeddings and similarity search with Pinecone"""

    def __init__(self):
        self.model = None
        self.index = None
        self.initialized = False

    def initialize(self):
        """Initialize embedding model and Pinecone"""
        try:
            model_name = current_app.config.get(
                "EMBEDDING_MODEL", "all-MiniLM-L6-v2"
            )

            # Load embedding model
            self.model = SentenceTransformer(model_name)

            api_key = current_app.config.get("PINECONE_API_KEY")
            index_name = current_app.config.get("PINECONE_INDEX_NAME")

            if not api_key or not index_name:
                logger.warning(
                    "PINECONE_API_KEY or PINECONE_INDEX_NAME missing -> Pinecone disabled"
                )
                self.index = None
                self.initialized = True
                return

            # Initialize Pinecone client (NEW API)
            pc = Pinecone(api_key=api_key)

            # Check if index exists (be defensive and log available indexes)
            try:
                index_list = pc.list_indexes()
                indexes = (
                    index_list.names()
                    if hasattr(index_list, "names")
                    else list(index_list)
                )
            except Exception as e:
                logger.error(f"Failed to list Pinecone indexes: {e}")
                indexes = []

            logger.info(f"Resolved Pinecone index name from config: '{index_name}'")
            logger.info(f"Pinecone indexes available: {indexes}")

            chosen_index = None
            if index_name in indexes:
                chosen_index = index_name
            else:
                # try to find a close match (contains / startswith)
                candidates = [i for i in indexes if index_name.lower() in i.lower() or i.lower().startswith(index_name.lower())]
                if candidates:
                    chosen_index = candidates[0]
                    logger.warning(
                        f"Pinecone index '{index_name}' not found; falling back to similar index '{chosen_index}'"
                    )
                elif indexes:
                    chosen_index = indexes[0]
                    logger.warning(
                        f"Pinecone index '{index_name}' not found; falling back to first available index '{chosen_index}'"
                    )
                else:
                    logger.error(f"Pinecone index '{index_name}' does not exist and no indexes available")

            if chosen_index:
                try:
                    self.index = pc.Index(chosen_index)
                    logger.info(f"Pinecone connected to index: {chosen_index}")
                except Exception as e:
                    logger.error(f"Failed to create index client for '{chosen_index}': {e}")
                    self.index = None
            else:
                self.index = None

            self.initialized = True

            logger.info(
                "Vector service initialized (model loaded, Pinecone available=%s)",
                "yes" if self.index else "no",
            )

        except Exception as e:
            logger.error(f"Failed to initialize vector service: {str(e)}")
            self.index = None
            self.initialized = True

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for given text"""
        if not self.initialized:
            self.initialize()

        try:
            embedding = self.model.encode(text)

            if hasattr(embedding, "tolist"):
                return embedding.tolist()

            return list(embedding)

        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            raise

    def upsert_product_embedding(
        self, product_id: str, text: str, metadata: Dict[str, Any] = None
    ):
        """Store product embedding in Pinecone"""
        if not self.initialized:
            self.initialize()

        if not self.index:
            logger.warning(
                "Pinecone index not available; skipping upsert for %s",
                product_id,
            )
            return

        try:
            embedding = self.generate_embedding(text)

            vector_data = {
                "id": product_id,
                "values": embedding,
                "metadata": metadata or {},
            }

            self.index.upsert(vectors=[vector_data])

            logger.info(f"Upserted embedding for product: {product_id}")

        except Exception as e:
            logger.error(f"Failed to upsert product embedding: {str(e)}")
            raise

    def search_similar_products(
        self,
        query_text: str,
        top_k: int = 10,
        filter_dict: Dict[str, Any] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar products using vector similarity"""
        if not self.initialized:
            self.initialize()

        logger.debug(f"search_similar_products called; initialized={self.initialized}, index_present={bool(self.index)}")
        if not self.index:
            logger.warning(
                "Pinecone index not available; semantic search disabled"
            )
            return []

        try:
            query_embedding = self.generate_embedding(query_text)

            logger.debug(f"Query embedding length: {len(query_embedding) if query_embedding else 0}")

            search_kwargs = {
                "vector": query_embedding,
                "top_k": top_k,
                "include_metadata": True,
                "include_values": False,
            }

            logger.debug(f"Pinecone query kwargs: top_k={top_k}, filter_provided={bool(filter_dict)}")

            if filter_dict:
                search_kwargs["filter"] = filter_dict

            results = self.index.query(**search_kwargs)
            logger.debug(f"Raw pinecone query response: {repr(results)[:1000]}")

            similar_products = []

            for match in results.matches:
                similar_products.append(
                    {
                        "id": match.id,
                        "score": match.score,
                        "metadata": match.metadata or {},
                    }
                )

            logger.debug(f"Similar products found: {len(similar_products)}")

            logger.info(
                f"Found {len(similar_products)} similar products for query: {query_text}"
            )

            return similar_products

        except Exception as e:
            logger.error(f"Failed to search similar products: {str(e)}")
            return []

    def delete_product_embedding(self, product_id: str):
        """Delete product embedding from Pinecone"""
        if not self.initialized:
            self.initialize()

        if not self.index:
            logger.warning(
                "Pinecone index not available; skip delete for %s",
                product_id,
            )
            return

        try:
            self.index.delete(ids=[product_id])

            logger.info(f"Deleted embedding for product: {product_id}")

        except Exception as e:
            logger.error(f"Failed to delete product embedding: {str(e)}")
            raise

    def get_index_stats(self) -> Dict[str, Any]:
        """Get Pinecone index statistics"""
        if not self.initialized:
            self.initialize()

        if not self.index:
            logger.warning(
                "Pinecone index not available; returning empty stats"
            )
            return {}

        try:
            stats = self.index.describe_index_stats()
            return stats

        except Exception as e:
            logger.error(f"Failed to get index stats: {str(e)}")
            return {}

    def batch_upsert_products(
        self,
        products: List[Dict[str, Any]],
        batch_size: int = 100,
    ):
        """Batch upsert multiple product embeddings"""
        if not self.initialized:
            self.initialize()

        if not self.index:
            logger.warning(
                "Pinecone index not available; skipping batch upsert"
            )
            return

        try:
            vectors = []

            for product in products:
                embedding = self.generate_embedding(product["text"])

                vectors.append(
                    {
                        "id": product["id"],
                        "values": embedding,
                        "metadata": product.get("metadata", {}),
                    }
                )

                if len(vectors) >= batch_size:
                    self.index.upsert(vectors=vectors)
                    vectors = []

            if vectors:
                self.index.upsert(vectors=vectors)

            logger.info(
                f"Batch upserted {len(products)} product embeddings"
            )

        except Exception as e:
            logger.error(f"Failed to batch upsert products: {str(e)}")
            raise