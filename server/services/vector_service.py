import logging
from typing import Any, Dict, List

from flask import current_app
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class VectorService:
    """Service for managing vector embeddings and similarity search with Pinecone"""

    def __init__(self):
        self.model = None
        self.index = None
        self.initialized = False

    def initialize(self):
        """Initialize Pinecone and embedding model. If Pinecone API key is missing,
        load only the embedding model and operate in local-only mode.
        """
        try:
            model_name = current_app.config.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

            # Load embedding model first (works without Pinecone)
            try:
                self.model = SentenceTransformer(model_name)
            except Exception as me:
                logger.error(f"Failed to load embedding model '{model_name}': {me}")
                raise

            api_key = current_app.config.get("PINECONE_API_KEY")
            # allow alternate env var names and capture which one was used
            index_name = current_app.config.get("PINECONE_INDEX_NAME")
            index_env_source = "PINECONE_INDEX_NAME"
            if not index_name:
                index_name = current_app.config.get("PINECONE_INDEX")
                index_env_source = "PINECONE_INDEX"

            if not api_key or not index_name:
                logger.warning(
                    "PINECONE_API_KEY or PINECONE_INDEX_NAME not set; vector service running without Pinecone index"
                )
                self.index = None
                self.initialized = True
                return

            # Lazy import pinecone and initialize index. Try multiple client APIs (robust detection).
            try:
                import pinecone

                index_obj = None
                tried_clients = []

                # Try several possible client class names exposed by different pinecone versions
                for cls_name in ("Pinecone", "PineconeClient", "Client"):
                    ClientCls = getattr(pinecone, cls_name, None)
                    if ClientCls:
                        tried_clients.append(cls_name)
                        try:
                            pc = ClientCls(api_key=api_key, environment=current_app.config.get("PINECONE_ENVIRONMENT"))
                            # many clients expose Index() factory
                            index_obj = getattr(pc, "Index")(index_name)
                            logger.info("Initialized Pinecone using client class '%s'", cls_name)
                            break
                        except Exception as ex_client:
                            logger.debug("Client class '%s' present but failed to init: %s", cls_name, ex_client)

                # If not created yet, try importing Pinecone directly (some installs use from pinecone import Pinecone)
                if index_obj is None:
                    try:
                        from pinecone import Pinecone as PineconeFromModule  # type: ignore

                        tried_clients.append("pinecone.Pinecone")
                        pc = PineconeFromModule(api_key=api_key, environment=current_app.config.get("PINECONE_ENVIRONMENT"))
                        index_obj = getattr(pc, "Index")(index_name)
                        logger.info("Initialized Pinecone using 'from pinecone import Pinecone'")
                    except Exception as ex_mod:
                        logger.debug("from pinecone import Pinecone failed: %s", ex_mod)

                # Fallback to legacy pinecone.init() if available
                if index_obj is None:
                    init_fn = getattr(pinecone, "init", None)
                    if callable(init_fn):
                        tried_clients.append("pinecone.init")
                        try:
                            pinecone.init(api_key=api_key, environment=current_app.config.get("PINECONE_ENVIRONMENT"))
                            index_obj = pinecone.Index(index_name)
                            logger.info("Initialized Pinecone using legacy pinecone.init()")
                        except Exception as ex_init:
                            logger.debug("pinecone.init() failed: %s", ex_init)

                if index_obj is None:
                    logger.error(
                        "Failed to initialize Pinecone index '%s' with any known client. Tried: %s",
                        index_name,
                        ",".join(tried_clients) or "(none)",
                    )
                    self.index = None
                else:
                    # Adapter to provide a stable interface expected by the rest of the code
                    class _IndexAdapter:
                        def __init__(self, idx):
                            self._idx = idx

                        def upsert(self, items):
                            try:
                                return self._idx.upsert(vectors=items)
                            except TypeError:
                                return self._idx.upsert(items)

                        def query(self, **kwargs):
                            try:
                                return self._idx.query(**kwargs)
                            except TypeError:
                                return self._idx.query(kwargs.get("vector"), kwargs.get("top_k"))

                        def delete(self, ids=None):
                            return self._idx.delete(ids=ids)

                        def describe_index_stats(self):
                            return self._idx.describe_index_stats()

                    self.index = _IndexAdapter(index_obj)
            except Exception as pe:
                logger.error(f"Failed to initialize Pinecone index '{index_name}': {pe}")
                # keep model loaded but do not raise to avoid crashing app on startup
                self.index = None

            self.initialized = True
            logger.info(
                "Vector service initialized (model loaded, Pinecone index=%s, source=%s, available=%s)",
                index_name,
                index_env_source,
                "yes" if self.index else "no",
            )

        except Exception as e:
            logger.error(f"Failed to initialize vector service: {str(e)}")
            raise

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for given text"""
        if not self.initialized:
            self.initialize()

        try:
            embedding = self.model.encode(text)
            return embedding.tolist()
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
            logger.warning("Pinecone index not available; skipping upsert for %s", product_id)
            return

        try:
            embedding = self.generate_embedding(text)

            vector_data = {
                "id": product_id,
                "values": embedding,
                "metadata": metadata or {},
            }

            self.index.upsert([vector_data])
            logger.info(f"Upserted embedding for product: {product_id}")

        except Exception as e:
            logger.error(f"Failed to upsert product embedding: {str(e)}")
            raise

    def search_similar_products(
        self, query_text: str, top_k: int = 10, filter_dict: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar products using vector similarity"""
        if not self.initialized:
            self.initialize()

        if not self.index:
            logger.warning("Pinecone index not available; semantic search disabled")
            return []

        try:
            query_embedding = self.generate_embedding(query_text)

            search_kwargs = {
                "vector": query_embedding,
                "top_k": top_k,
                "include_metadata": True,
                "include_values": False,
            }

            if filter_dict:
                search_kwargs["filter"] = filter_dict

            results = self.index.query(**search_kwargs)

            similar_products = []
            for match in results.get("matches", []):
                similar_products.append(
                    {
                        "id": match.get("id"),
                        "score": match.get("score"),
                        "metadata": match.get("metadata", {}),
                    }
                )

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
            logger.warning("Pinecone index not available; skip delete for %s", product_id)
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
            logger.warning("Pinecone index not available; returning empty stats")
            return {}

        try:
            stats = self.index.describe_index_stats()
            return stats
        except Exception as e:
            logger.error(f"Failed to get index stats: {str(e)}")
            return {}

    def batch_upsert_products(
        self, products: List[Dict[str, Any]], batch_size: int = 100
    ):
        """Batch upsert multiple product embeddings"""
        if not self.initialized:
            self.initialize()

        if not self.index:
            logger.warning("Pinecone index not available; skipping batch upsert")
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
                    self.index.upsert(vectors)
                    vectors = []

            if vectors:
                self.index.upsert(vectors)

            logger.info(f"Batch upserted {len(products)} product embeddings")

        except Exception as e:
            logger.error(f"Failed to batch upsert products: {str(e)}")
            raise
