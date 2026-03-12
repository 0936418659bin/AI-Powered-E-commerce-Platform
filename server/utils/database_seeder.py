import json
import logging
import uuid
from datetime import datetime

from models.product import Product
from services.product_service import ProductService

logger = logging.getLogger(__name__)


class DatabaseSeeder:
    """Utility class for seeding the database with initial data"""

    def __init__(self, db):
        self.db = db
        self.product_service = ProductService()

    def seed_products(self):
        """Seed the database with sample products"""
        try:
            if Product.query.count() > 0:
                logger.info("Products already exist, skipping seeding")
                return

            products_data = self._get_sample_products()

            logger.info(f"Seeding {len(products_data)} products...")

            for product_data in products_data:
                try:
                    product = Product(**product_data)
                    self.db.session.add(product)
                    self.db.session.flush()

                    search_text = product.get_search_text()
                    metadata = {
                        "category": product.category,
                        "subcategory": product.subcategory,
                        "brand": product.brand,
                        "price": product.price,
                        "rating": product.rating,
                        "in_stock": product.is_in_stock(),
                    }

                    self.product_service.vector_service.upsert_product_embedding(
                        product.id, search_text, metadata
                    )

                    product.embedding_id = product.id

                except Exception as e:
                    logger.error(
                        f"Error seeding product {product_data.get('name', 'Unknown')}: {str(e)}"
                    )
                    continue

            self.db.session.commit()
            logger.info("Products seeded successfully")

        except Exception as e:
            logger.error(f"Error seeding products: {str(e)}")
            self.db.session.rollback()
            raise

    def _get_sample_products(self):
        """Get sample product data"""
        return [
            {
                "id": str(uuid.uuid4()),
                "name": "iPhone 15 Pro",
                "description": "Latest iPhone with titanium design, A17 Pro chip, and advanced camera system",
                "price": 999.0,
                "original_price": 1099.0,
                "category": "Electronics",
                "subcategory": "Smartphones",
                "brand": "Apple",
                "rating": 4.8,
                "review_count": 2847,
                "image_url": "https://images.pexels.com/photos/18525574/pexels-photo-18525574.jpeg",
                "stock": 45,
                "features": [
                    "A17 Pro Chip",
                    '6.1" Display',
                    "48MP Camera",
                    "Titanium Build",
                ],
                "is_on_sale": True,
                "sale_percentage": 9,
            },
            {
                "id": str(uuid.uuid4()),
                "name": "Samsung Galaxy S24 Ultra",
                "description": "Premium Android flagship with S Pen, advanced AI features, and 200MP camera",
                "price": 1199.0,
                "category": "Electronics",
                "subcategory": "Smartphones",
                "brand": "Samsung",
                "rating": 4.7,
                "review_count": 1923,
                "image_url": "https://images.pexels.com/photos/30466741/pexels-photo-30466741.jpeg",
                "stock": 32,
                "features": ["200MP Camera", "S Pen", '6.8" Display', "AI Features"],
            },
            {
                "id": str(uuid.uuid4()),
                "name": "Google Pixel 8 Pro",
                "description": "AI-powered photography, pure Android experience, and advanced computational features",
                "price": 899.0,
                "category": "Electronics",
                "subcategory": "Smartphones",
                "brand": "Google",
                "rating": 4.6,
                "review_count": 1456,
                "image_url": "https://images.pexels.com/photos/32218867/pexels-photo-32218867.jpeg",
                "stock": 28,
                "features": [
                    "AI Photography",
                    "Pure Android",
                    '6.7" Display',
                    "Titan M2 Chip",
                ],
            },
            {
                "id": str(uuid.uuid4()),
                "name": 'MacBook Pro 16" M3 Max',
                "description": "Ultimate creative powerhouse with M3 Max chip, stunning Liquid Retina XDR display",
                "price": 2499.0,
                "original_price": 2699.0,
                "category": "Electronics",
                "subcategory": "Laptops",
                "brand": "Apple",
                "rating": 4.9,
                "review_count": 987,
                "image_url": "https://images.pexels.com/photos/18105/pexels-photo.jpg",
                "stock": 15,
                "features": [
                    "M3 Max Chip",
                    '16" Liquid Retina XDR',
                    "36GB RAM",
                    "1TB SSD",
                ],
                "is_on_sale": True,
                "sale_percentage": 7,
            },
            {
                "id": str(uuid.uuid4()),
                "name": "Dell XPS 13 Plus",
                "description": "Ultra-premium Windows laptop with stunning OLED display and sleek design",
                "price": 1399.0,
                "category": "Electronics",
                "subcategory": "Laptops",
                "brand": "Dell",
                "rating": 4.5,
                "review_count": 743,
                "image_url": "https://images.pexels.com/photos/3776438/pexels-photo-3776438.jpeg",
                "stock": 22,
                "features": [
                    '13.4" OLED',
                    "Intel 12th Gen",
                    "16GB RAM",
                    "Premium Build",
                ],
            },
            {
                "id": str(uuid.uuid4()),
                "name": "ASUS ROG Zephyrus G16",
                "description": "High-performance gaming laptop with RTX 4070, AMD Ryzen 9, and 240Hz display",
                "price": 1899.0,
                "category": "Electronics",
                "subcategory": "Gaming Laptops",
                "brand": "ASUS",
                "rating": 4.7,
                "review_count": 634,
                "image_url": "https://images.pexels.com/photos/12877878/pexels-photo-12877878.jpeg",
                "stock": 18,
                "features": ["RTX 4070", "AMD Ryzen 9", "240Hz Display", "ROG Design"],
            },
            {
                "id": str(uuid.uuid4()),
                "name": "Sony WH-1000XM5",
                "description": "Industry-leading noise canceling headphones with exceptional sound quality",
                "price": 349.0,
                "original_price": 399.0,
                "category": "Electronics",
                "subcategory": "Headphones",
                "brand": "Sony",
                "rating": 4.8,
                "review_count": 3421,
                "image_url": "https://images.pexels.com/photos/10292805/pexels-photo-10292805.jpeg",
                "stock": 67,
                "features": [
                    "Active Noise Canceling",
                    "30hr Battery",
                    "Hi-Res Audio",
                    "Touch Controls",
                ],
                "is_on_sale": True,
                "sale_percentage": 13,
            },
            {
                "id": str(uuid.uuid4()),
                "name": "Apple AirPods Pro (2nd Gen)",
                "description": "Advanced active noise cancellation, spatial audio, and seamless Apple integration",
                "price": 249.0,
                "category": "Electronics",
                "subcategory": "Earbuds",
                "brand": "Apple",
                "rating": 4.7,
                "review_count": 2847,
                "image_url": "https://images.pexels.com/photos/3825517/pexels-photo-3825517.jpeg",
                "stock": 89,
                "features": [
                    "Active Noise Canceling",
                    "Spatial Audio",
                    "H2 Chip",
                    "MagSafe Case",
                ],
            },
            {
                "id": str(uuid.uuid4()),
                "name": "Bose QuietComfort Ultra",
                "description": "Premium comfort with world-class noise cancellation and immersive audio",
                "price": 429.0,
                "category": "Electronics",
                "subcategory": "Headphones",
                "brand": "Bose",
                "rating": 4.6,
                "review_count": 1234,
                "image_url": "https://images.pexels.com/photos/7748203/pexels-photo-7748203.jpeg",
                "stock": 34,
                "features": [
                    "QuietComfort Technology",
                    "Spatial Audio",
                    "Premium Materials",
                    "24hr Battery",
                ],
            },
            {
                "id": str(uuid.uuid4()),
                "name": "PlayStation 5 Slim",
                "description": "Latest PlayStation console with enhanced performance and sleeker design",
                "price": 499.0,
                "category": "Electronics",
                "subcategory": "Gaming Consoles",
                "brand": "Sony",
                "rating": 4.8,
                "review_count": 4567,
                "image_url": "https://images.pexels.com/photos/13189272/pexels-photo-13189272.jpeg",
                "stock": 12,
                "features": [
                    "Custom SSD",
                    "4K Gaming",
                    "Ray Tracing",
                    "DualSense Controller",
                ],
            },
            {
                "id": str(uuid.uuid4()),
                "name": "Xbox Series X",
                "description": "Most powerful Xbox ever with 4K gaming, ray tracing, and quick resume",
                "price": 499.0,
                "category": "Electronics",
                "subcategory": "Gaming Consoles",
                "brand": "Microsoft",
                "rating": 4.7,
                "review_count": 3456,
                "image_url": "https://images.pexels.com/photos/5626726/pexels-photo-5626726.jpeg",
                "stock": 8,
                "features": [
                    "12 TeraFLOPS",
                    "4K/120fps",
                    "Quick Resume",
                    "Smart Delivery",
                ],
            },
            {
                "id": str(uuid.uuid4()),
                "name": "Razer DeathAdder V3 Pro",
                "description": "Professional esports gaming mouse with 90-hour battery and precision sensors",
                "price": 149.0,
                "original_price": 179.0,
                "category": "Electronics",
                "subcategory": "Gaming Accessories",
                "brand": "Razer",
                "rating": 4.6,
                "review_count": 892,
                "image_url": "https://images.pexels.com/photos/8176505/pexels-photo-8176505.jpeg",
                "stock": 156,
                "features": [
                    "90hr Battery",
                    "Focus Pro Sensor",
                    "Ergonomic Design",
                    "RGB Lighting",
                ],
                "is_on_sale": True,
                "sale_percentage": 17,
            },
            {
                "id": str(uuid.uuid4()),
                "name": "Amazon Echo Studio",
                "description": "High-fidelity smart speaker with 3D audio and Dolby Atmos support",
                "price": 199.0,
                "category": "Electronics",
                "subcategory": "Smart Speakers",
                "brand": "Amazon",
                "rating": 4.4,
                "review_count": 2341,
                "image_url": "https://images.pexels.com/photos/4790268/pexels-photo-4790268.jpeg",
                "stock": 45,
                "features": [
                    "3D Audio",
                    "Dolby Atmos",
                    "Voice Control",
                    "Smart Home Hub",
                ],
            },
            {
                "id": str(uuid.uuid4()),
                "name": "Nest Thermostat",
                "description": "Smart thermostat that learns your schedule and saves energy automatically",
                "price": 249.0,
                "category": "Electronics",
                "subcategory": "Smart Home",
                "brand": "Google",
                "rating": 4.5,
                "review_count": 1876,
                "image_url": "https://images.pexels.com/photos/190048/pexels-photo-190048.jpeg",
                "stock": 78,
                "features": [
                    "Auto-Schedule",
                    "Energy Saving",
                    "Remote Control",
                    "Voice Commands",
                ],
            },
            {
                "id": str(uuid.uuid4()),
                "name": "Ring Video Doorbell Pro 2",
                "description": "Advanced smart doorbell with 3D motion detection and crystal clear HD video",
                "price": 279.0,
                "original_price": 329.0,
                "category": "Electronics",
                "subcategory": "Security",
                "brand": "Ring",
                "rating": 4.3,
                "review_count": 3421,
                "image_url": "https://images.pexels.com/photos/9461215/pexels-photo-9461215.jpeg",
                "stock": 23,
                "features": [
                    "3D Motion Detection",
                    "HD Video",
                    "Two-Way Talk",
                    "Night Vision",
                ],
                "is_on_sale": True,
                "sale_percentage": 15,
            },
        ]
