import os
import importlib
from dotenv import load_dotenv

# Load .env
load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME", "ecommerce-chatbot")

print("Checking Pinecone connection...\n")

if not api_key:
    print("❌ PINECONE_API_KEY not found in environment variables")
    exit(1)

try:
    pinecone = importlib.import_module("pinecone")

    # Try new Pinecone client first
    if hasattr(pinecone, "Pinecone"):
        try:
            pc = pinecone.Pinecone(api_key=api_key)
            li = pc.list_indexes()
            indexes = li.names() if hasattr(li, "names") else list(li)
            print("Using Pinecone class (new client)")
            print("Indexes discovered:", indexes)
            print("Configured index:", index_name)
            if index_name in indexes:
                print(f"✅ Index exists: {index_name}")
            else:
                print(f"❌ Index not found: {index_name}")
            raise SystemExit(0)
        except SystemExit:
            raise
        except Exception as e_new:
            print("⚠️ New Pinecone client check failed:", e_new)

    # Fallback to legacy pinecone.init() API
    if hasattr(pinecone, "init"):
        try:
            pinecone.init(api_key=api_key)
            indexes = pinecone.list_indexes()
            indexes = indexes.names() if hasattr(indexes, "names") else list(indexes)
            print("Using legacy pinecone.init() client")
            print("Indexes discovered:", indexes)
            print("Configured index:", index_name)
            if index_name in indexes:
                print(f"✅ Index exists: {index_name}")
            else:
                print(f"❌ Index not found: {index_name}")
            raise SystemExit(0)
        except SystemExit:
            raise
        except Exception as e_legacy:
            print("⚠️ Legacy Pinecone client check failed:", e_legacy)

    print("❌ Pinecone package does not expose a supported client API")

except Exception as e:
    print("❌ Failed to import or check Pinecone:")
    print(e)