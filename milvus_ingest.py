import json
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from sentence_transformers import SentenceTransformer
import os

# --- CONFIGURATION ---
DICTIONARY_PATH = "dictionary.json"
# MILVUS LITE: We use a local file instead of a host/port
MILVUS_URI = "./milvus_demo.db" 
COLLECTION_NAME = "sql_data_dictionary"

# Load Model
print("⏳ Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_collection():
    # Connect to local file
    connections.connect("default", uri=MILVUS_URI)
    
    if utility.has_collection(COLLECTION_NAME):
        return Collection(COLLECTION_NAME)
    
    # Create Collection if it doesn't exist
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="table_name", dtype=DataType.VARCHAR, max_length=200),
        # Storing the raw JSON schema allows the agent to read it later
        FieldSchema(name="schema_text", dtype=DataType.VARCHAR, max_length=6000), 
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)
    ]
    schema = CollectionSchema(fields, "Schema for SQL Data Dictionary")
    collection = Collection(COLLECTION_NAME, schema)
    
    # Create Index for fast retrieval
    # Note: Milvus Lite supports IVF_FLAT or FLAT
    index_params = {"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
    collection.create_index("embedding", index_params)
    return collection

def upsert_dictionary():
    """
    Handles INSERT and UPDATE logic.
    """
    collection = get_collection()
    
    try:
        with open(DICTIONARY_PATH, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("❌ dictionary.json not found.")
        return

    if not data.get("tables"):
        print("⚠️ No tables found in JSON.")
        return

    # 1. Identify tables to process
    table_names_to_process = [t['name'] for t in data["tables"]]
    
    # 2. Delete existing records for these tables (Clean Update)
    if collection.num_entities > 0:
        collection.load()
        # Escaping quotes for the expression
        names_formatted = [f'"{name}"' for name in table_names_to_process]
        expr = f"table_name in [{', '.join(names_formatted)}]"
        
        # Note: Delete operations in Milvus might take a moment to reflect in search
        collection.delete(expr)
        print(f"♻️  Cleaned old versions for: {table_names_to_process}")

    # 3. Prepare new data
    names = []
    schemas = []
    vectors = []

    print("🚀 Generating embeddings...")
    for table in data["tables"]:
        col_names = ", ".join([c['name'] for c in table['columns']])
        # Create rich text for semantic search
        text_to_embed = f"Table: {table['name']}. Description: {table['description']}. Columns: {col_names}"
        
        names.append(table['name'])
        schemas.append(json.dumps(table)) # Store the raw JSON schema
        vectors.append(model.encode(text_to_embed))

    # 4. Insert Data
    collection.insert([names, schemas, vectors])
    collection.flush()
    
    # 5. Re-index (Good practice after updates)
    index_params = {"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
    collection.create_index("embedding", index_params)
    
    print(f"✅ Successfully upserted {len(names)} tables.")
    print(f"📊 Total Entities in DB: {collection.num_entities}")

if __name__ == "__main__":
    upsert_dictionary()
