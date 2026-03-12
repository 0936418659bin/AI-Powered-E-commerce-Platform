import pinecone

print("pinecone file:", getattr(pinecone, "__file__", None))
print("pinecone version:", getattr(pinecone, "__version__", None))
print("has Pinecone class:", hasattr(pinecone, "Pinecone"))
print("has init():", hasattr(pinecone, "init"))

print("list dir:")
for n in dir(pinecone):
    if n.lower().startswith("list") or n.lower().startswith("index"):
        print(n)