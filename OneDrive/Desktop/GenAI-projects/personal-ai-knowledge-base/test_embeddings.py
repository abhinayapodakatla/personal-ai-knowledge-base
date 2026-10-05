from sentence_transformers import SentenceTransformer


# Load embedding model
print("🧠 Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("✅ Model loaded!")


# Test sentences
sentence1 = "Machine Learning learns patterns from data."

sentence2 = "ML allows computers to learn from data."


# Create embeddings
embedding1 = model.encode(sentence1)

embedding2 = model.encode(sentence2)


# Display results
print("\nSentence 1:")
print(sentence1)

print("\nEmbedding 1:")
print(embedding1)


print("\nSentence 2:")
print(sentence2)

print("\nEmbedding 2:")
print(embedding2)


# Show vector size
print("\n📏 Embedding size:")
print(len(embedding1))