import json
import requests
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sentence_transformers import SentenceTransformer, util
from sklearn.manifold import TSNE
from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType

# --- CONFIGURATION ---
MODEL_NAME = "all-MiniLM-L6-v2"
DICTIONARY_PATH = "dictionary.json"
MILVUS_URI = "./milvus_demo.db"
COLLECTION_NAME = "sql_data_dictionary"

# Ollama Configuration
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
LLAMA_MODEL = "llama3.1:8b"

st.set_page_config(page_title="Schema Intelligence", layout="wide")

# --- LOAD RESOURCES ---
@st.cache_resource
def load_resources():
    print("⏳ Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)
    return model

@st.cache_data
def load_visualization_data():
    """
    Loads data specifically for the Tab 1 Visualization (DataFrame format).
    """
    try:
        with open(DICTIONARY_PATH, "r") as f:
            data = json.load(f)
        
        rows = []
        for table in data["tables"]:
            table_name = table["name"]
            rows.append({
                "type": "Table",
                "name": table_name,
                "text": f"Table: {table_name}. {table.get('description', '')}",
                "display_text": f"<b>Table: {table_name}</b><br>{table.get('description', '')}"
            })
            for col in table["columns"]:
                rows.append({
                    "type": "Column",
                    "name": f"{table_name}.{col['name']}",
                    "text": f"Column: {col['name']} in Table: {table_name}. Meaning: {col.get('meaning', '')}",
                    "display_text": f"<b>{table_name}.{col['name']}</b><br>{col.get('meaning', '')}"
                })
        return pd.DataFrame(rows)
    except FileNotFoundError:
        return pd.DataFrame()

def load_raw_dictionary():
    """Loads the raw JSON for editing/saving in Tab 2."""
    try:
        with open(DICTIONARY_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"tables": []}

def save_raw_dictionary(data):
    """Saves the updated dictionary to JSON."""
    with open(DICTIONARY_PATH, "w") as f:
        json.dump(data, f, indent=4)

def rebuild_milvus_db(full_data, model):
    """
    Wipes and recreates the Milvus collection with the updated data.
    Ensures external agents see the new descriptions.
    """
    try:
        connections.connect("default", uri=MILVUS_URI)
        
        # 1. Drop Old Collection
        if utility.has_collection(COLLECTION_NAME):
            utility.drop_collection(COLLECTION_NAME)
            
        # 2. Define Schema (Column-Level Granularity for precision)
        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=6000), # The searchable text
            FieldSchema(name="table_name", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="schema_text", dtype=DataType.VARCHAR, max_length=6000), # Full JSON context
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)
        ]
        schema = CollectionSchema(fields, "Schema Dictionary")
        collection = Collection(COLLECTION_NAME, schema)
        
        # 3. Prepare Data
        texts = []
        table_names = []
        schema_texts = [] # We store the column JSON here
        
        # Flatten columns into vectors
        for t in full_data["tables"]:
            for c in t["columns"]:
                # The text we embed for searching
                semantic_text = f"Table: {t['name']}, Column: {c['name']}, Type: {c.get('type','')}, Meaning: {c.get('meaning','')}"
                
                texts.append(semantic_text)
                table_names.append(t["name"])
                # Store metadata as JSON string
                schema_texts.append(json.dumps({
                    "name": t["name"],
                    "columns": [c] # Store single column context
                }))

        # 4. Embed and Insert
        if texts:
            embeddings = model.encode(texts)
            entities = [
                texts,
                table_names,
                schema_texts,
                embeddings
            ]
            collection.insert(entities)
            
            # 5. Index
            index_params = {"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
            collection.create_index(field_name="embedding", index_params=index_params)
            collection.load()
            return True
    except Exception as e:
        st.error(f"Milvus Rebuild Error: {str(e)}")
        return False
    return True

@st.cache_data
def get_all_columns_flattened():
    """
    Returns a list of all columns in the DB with their current metadata.
    """
    data = load_raw_dictionary()
    cols = []
    for table in data["tables"]:
        t_name = table["name"]
        for col in table["columns"]:
            cols.append({
                "full_name": f"{t_name}.{col['name']}",
                "table": t_name,
                "column": col['name'],
                "meaning": col.get("meaning", ""),
                "type": col.get("type", "TEXT")
            })
    return cols

# --- HELPER: RANKING ---
def get_ranked_schema(query, model):
    """
    Retrieves ALL columns, scores them against the query, and returns them sorted.
    """
    all_cols = get_all_columns_flattened()
    
    # Construct text for embedding
    texts = [
        f"Table: {c['table']}, Column: {c['column']}, Type: {c['type']}, Meaning: {c['meaning']}"
        for c in all_cols
    ]
    
    # Encode Query and All Columns
    query_vec = model.encode(query)
    col_vecs = model.encode(texts)
    
    # Calculate Similarity
    scores = util.cos_sim(query_vec, col_vecs)[0]
    
    # Attach scores
    for idx, score in enumerate(scores):
        all_cols[idx]["score"] = score.item()
        
    # Sort by Score Descending
    sorted_cols = sorted(all_cols, key=lambda x: x["score"], reverse=True)
    
    # Add Rank
    for rank, item in enumerate(sorted_cols, 1):
        item["rank"] = rank
        
    return sorted_cols

# --- HELPER: LLAMA SQL ---
def generate_sql_with_reasoning(query, schema_context):
    context_str = "\n".join([
        f"- {item['table']}.{item['column']} ({item['type']}): {item['meaning']}"
        for item in schema_context
    ])
    
    system_prompt = f"""You are a SQL expert.
    1. First, explain your REASONING. Why did you pick specific columns? 
    2. Then, provide the SQL query.
    3. Use this schema context:
    {context_str}
    
    Format your response exactly like this:
    REASONING:
    (Your explanation here)
    
    SQL:
    ```sql
    (Your SQL here)
    ```
    """
    
    payload = {
        "model": LLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        "stream": False,
        "options": {"temperature": 0}
    }
    
    try:
        response = requests.post(OLLAMA_CHAT_URL, json=payload)
        if response.status_code == 200:
            return response.json().get("message", {}).get("content", "No response.")
        return f"Error: {response.text}"
    except Exception as e:
        return f"Connection Error: {str(e)}"

# --- INITIALIZE ---
if 'generated_result' not in st.session_state:
    st.session_state.generated_result = None
if 'current_query' not in st.session_state:
    st.session_state.current_query = ""

model = load_resources()

st.title("‍♂️ Schema Intelligence & Optimization")

# --- TABS ---
tab1, tab2 = st.tabs(["🗺️ Global Semantic Map", "🛠️ SQL & Schema Optimizer"])

# --- TAB 1: VISUALIZATION ---
with tab1:
    st.subheader("2D Visualization of Schema Vectors")
    df_viz = load_visualization_data()
    
    if not df_viz.empty:
        col_controls, col_map = st.columns([1, 4])
        
        with col_controls:
            if st.button("Generate/Refresh Map"):
                with st.spinner("Generating embeddings and running t-SNE..."):
                    embeddings = model.encode(df_viz["text"].tolist())
                    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(df_viz)-1))
                    projections = tsne.fit_transform(embeddings)
                    df_viz["x"] = projections[:, 0]
                    df_viz["y"] = projections[:, 1]
                    st.session_state.viz_data = df_viz
            
        with col_map:
            if 'viz_data' in st.session_state:
                fig = px.scatter(
                    st.session_state.viz_data, 
                    x="x", y="y", color="type",
                    hover_data={"x": False, "y": False, "name": True, "display_text": False},
                    custom_data=["display_text"],
                    title="Schema Semantic Proximity Map"
                )
                fig.update_traces(hovertemplate="%{customdata[0]}<extra></extra>")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("Click 'Generate Map' to see the visualization.")
    else:
        st.warning("No data found in dictionary.json")

# --- TAB 2: OPTIMIZER ---
with tab2:
    st.markdown("### Interactive SQL Generation & Tuning")

    # 1. INPUT
    query_input = st.text_input("Enter your natural language question:", value=st.session_state.current_query)

    if st.button("🚀 Generate SQL"):
        st.session_state.current_query = query_input
        
        with st.spinner("Analyzing schema and generating SQL..."):
            ranked_cols = get_ranked_schema(query_input, model)
            top_k = ranked_cols[:10]
            llm_response = generate_sql_with_reasoning(query_input, top_k)
            
            st.session_state.generated_result = {
                "ranked_cols": ranked_cols,
                "top_k": top_k,
                "llm_response": llm_response
            }

    # 2. RESULTS DISPLAY
    if st.session_state.generated_result:
        res = st.session_state.generated_result
        
        # Parse Response
        raw_text = res["llm_response"]
        reasoning = "No reasoning provided."
        sql_code = "-- No SQL generated"
        
        if "REASONING:" in raw_text and "SQL:" in raw_text:
            parts = raw_text.split("SQL:")
            reasoning = parts[0].replace("REASONING:", "").strip()
            sql_code = parts[1].strip().replace("```sql", "").replace("```", "")
        elif "```sql" in raw_text:
            sql_code = raw_text.split("```sql")[1].split("```")[0]
            reasoning = raw_text.split("```sql")[0]

        # UI Layout
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("🤖 Model Output")
            st.info(reasoning)
            st.code(sql_code, language="sql")
        with col2:
            st.subheader("📚 Context Provided (Top 10)")
            df_context = pd.DataFrame(res["top_k"])
            st.dataframe(df_context[["rank", "score", "full_name", "meaning"]], use_container_width=True)

        st.divider()
        
        # 3. INTERACTIVE DEBUGGING
        st.subheader("🕵️ Debugger: Why wasn't my column picked?")
        
        all_col_names = [c["full_name"] for c in res["ranked_cols"]]
        target_col_name = st.selectbox("Search for a missing column:", options=all_col_names)
        
        target_col = next((c for c in res["ranked_cols"] if c["full_name"] == target_col_name), None)
        
        if target_col:
            # Metrics
            curr_score = target_col["score"]
            curr_rank = target_col["rank"]
            cutoff_score = res["top_k"][-1]["score"]
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Current Rank", f"#{curr_rank}")
            c2.metric("Current Score", f"{curr_score:.4f}")
            c3.metric("Top 10 Cutoff", f"{cutoff_score:.4f}")
            
            if curr_rank <= 10:
                st.success("✅ Passed to LLM.")
            else:
                st.error(f"❌ Missed Cutoff by {(cutoff_score - curr_score):.4f}")

            # 4. OPTIMIZATION LOOP
            st.markdown("#### ⚡ Improve Ranking")
            
            col_edit, col_sim = st.columns([2, 1])
            with col_edit:
                new_desc = st.text_area("New Description:", value=target_col["meaning"], height=100)
                
            with col_sim:
                if st.button("🧪 Simulate New Score"):
                    temp_text = f"Table: {target_col['table']}, Column: {target_col['column']}, Type: {target_col['type']}, Meaning: {new_desc}"
                    new_vec = model.encode(temp_text)
                    query_vec = model.encode(st.session_state.current_query)
                    new_score = util.cos_sim(query_vec, new_vec).item()
                    
                    better_than = [c for c in res["ranked_cols"] if c["score"] > new_score]
                    new_rank = len(better_than) + 1
                    
                    st.metric("Projected Score", f"{new_score:.4f}", delta=f"{new_score - curr_score:.4f}")
                    
                    if new_rank <= 10:
                        st.success(f"Projected Rank: #{new_rank} (Pass!)")
                    else:
                        st.warning(f"Projected Rank: #{new_rank} (Fail)")
                        
                if st.button("💾 Save & Refresh"):
                    # 1. Load & Update JSON
                    full_data = load_raw_dictionary()
                    updated = False
                    for t in full_data["tables"]:
                        if t["name"] == target_col["table"]:
                            for c in t["columns"]:
                                if c["name"] == target_col["column"]:
                                    c["meaning"] = new_desc
                                    updated = True
                                    break
                    
                    if updated:
                        save_raw_dictionary(full_data)
                        
                        # 2. CRITICAL: Clear Cache & Rebuild DB
                        st.cache_data.clear()
                        with st.spinner("Re-indexing Milvus Database..."):
                            rebuild_milvus_db(full_data, model)
                        
                        st.toast("Updated JSON & Milvus! Reloading...", icon="✅")
                        st.rerun()
                    else:
                        st.error("Column not found in JSON.")
