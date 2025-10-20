import os
import time
import json
import mysql.connector
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text
from typing import Optional, Dict, List, Tuple

# --- Load config ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", 3306))
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

if not GEMINI_API_KEY:
    raise RuntimeError("Please set GEMINI_API_KEY in your .env file")

# --- Configure OpenAI (Gemini endpoint) ---
client = OpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
GEMINI_MODEL = "gemini-2.0-flash-exp"

# --- SQL / DB helpers ---
def get_sqlalchemy_engine():
    """Create SQLAlchemy engine with proper connection pooling"""
    uri = f"mysql+mysqlconnector://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"
    return create_engine(uri, pool_pre_ping=True, pool_recycle=3600)

SCHEMA_QUERY = """
SELECT table_name AS `TABLE NAME`,
       column_name AS `COLUMN NAME`,
       data_type AS `DATA TYPE`,
       character_maximum_length AS `MAX LENGTH`,
       is_nullable AS `IS NULLABLE`,
       column_key AS `COLUMN KEY`,
       column_default AS `DEFAULT VALUE`,
       extra AS `EXTRA INFO`
FROM information_schema.columns
WHERE table_schema = :schema
ORDER BY table_name, ordinal_position;
"""

def fetch_schema_rows() -> List[Dict]:
    """Fetch database schema information"""
    engine = get_sqlalchemy_engine()
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(SCHEMA_QUERY), conn, params={"schema": MYSQL_DATABASE})
        rows = df.fillna("").to_dict(orient="records")
        return rows
    except Exception as e:
        print(f"Error fetching schema: {e}")
        raise

# --- Embedding model ---
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for texts"""
    return embed_model.encode(
        texts, 
        show_progress_bar=False, 
        normalize_embeddings=True
    ).tolist()

# --- Chroma setup ---
def initialize_chroma():
    """Initialize ChromaDB client with proper error handling"""
    global chroma_client, schema_coll, cache_coll
    
    try:
        # Create persistent directory if it doesn't exist
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        
        # Try modern ChromaDB initialization (v0.4.0+)
        chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        print(f"ChromaDB initialized at: {CHROMA_PERSIST_DIR}")
        
    except Exception as e:
        print(f"Warning: Could not use PersistentClient: {e}")
        print("Falling back to ephemeral client (data will not persist)")
        chroma_client = chromadb.Client()
    
    # Get or create collections
    try:
        schema_coll = chroma_client.get_or_create_collection(
            name="schema",
            metadata={"description": "Database schema metadata"}
        )
        cache_coll = chroma_client.get_or_create_collection(
            name="query_cache",
            metadata={"description": "Cached query results"}
        )
        print("Collections initialized successfully")
    except Exception as e:
        print(f"Error initializing collections: {e}")
        raise

# Initialize ChromaDB
initialize_chroma()

def build_document_text_from_row(row: Dict) -> str:
    """Build searchable document text from schema row"""
    parts = [
        f"TABLE: {row.get('TABLE NAME', '')}",
        f"COLUMN: {row.get('COLUMN NAME', '')}",
        f"TYPE: {row.get('DATA TYPE', '')}",
        f"MAX_LENGTH: {row.get('MAX LENGTH', '')}",
        f"IS_NULLABLE: {row.get('IS NULLABLE', '')}",
        f"COLUMN_KEY: {row.get('COLUMN KEY', '')}",
        f"DEFAULT: {row.get('DEFAULT VALUE', '')}",
        f"EXTRA: {row.get('EXTRA INFO', '')}",
    ]
    return " | ".join(str(p) for p in parts if p)

def populate_schema_collection(overwrite: bool = False):
    """Populate or update schema collection in ChromaDB"""
    print("Fetching database schema...")
    rows = fetch_schema_rows()
    print(f"Fetched {len(rows)} schema rows")
    
    if not rows:
        print("Warning: No schema rows found!")
        return
    
    texts = []
    metadatas = []
    ids = []
    
    for i, r in enumerate(rows):
        doc = build_document_text_from_row(r)
        texts.append(doc)
        metadatas.append({
            "table": str(r.get("TABLE NAME", "")),
            "column": str(r.get("COLUMN NAME", "")),
            "data_type": str(r.get("DATA TYPE", "")),
        })
        ids.append(f"schema_row_{i}_{r.get('TABLE NAME')}_{r.get('COLUMN NAME')}")
    
    embeddings = embed_texts(texts)
    
    if overwrite:
        print("Rebuilding schema collection...")
        try:
            chroma_client.delete_collection("schema")
        except Exception as e:
            print(f"Note: Could not delete collection (may not exist): {e}")
        
        global schema_coll
        schema_coll = chroma_client.create_collection(name="schema")
    
    try:
        # Use upsert to handle duplicates gracefully
        schema_coll.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings
        )
        print(f"✓ Schema collection {'rebuilt' if overwrite else 'updated'} with {len(texts)} items")
    except Exception as e:
        print(f"Error populating schema collection: {e}")
        raise

def query_cache_find_similar(
    question: str, 
    top_k: int = 1, 
    score_threshold: float = 0.80
) -> Tuple[Optional[Dict], float]:
    """Search cache for similar questions"""
    try:
        q_emb = embed_texts([question])[0]
        results = cache_coll.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["metadatas", "documents", "distances"]
        )
        
        if results and len(results["distances"][0]) > 0:
            distance = results["distances"][0][0]
            score = 1 - distance
            
            if score >= score_threshold:
                hit = {
                    "question": results["documents"][0][0],
                    "sql": results["metadatas"][0][0].get("sql", ""),
                    "answer": results["metadatas"][0][0].get("answer", ""),
                }
                return hit, score
        
        return None, 0.0
    except Exception as e:
        print(f"Error querying cache: {e}")
        return None, 0.0

def cache_query_result(question: str, sql: str, answer: str):
    """Cache a query result for future use"""
    try:
        q_emb = embed_texts([question])[0]
        item_id = f"qcache_{int(time.time() * 1000)}"
        
        cache_coll.upsert(
            ids=[item_id],
            documents=[question],
            metadatas=[{"sql": sql, "answer": answer}],
            embeddings=[q_emb]
        )
        print("Query result cached successfully")
    except Exception as e:
        print(f"Warning: Could not cache result: {e}")

# --- Gemini SQL Generation ---
GEN_SQL_PROMPT_TEMPLATE = """
You are an expert SQL assistant for a MySQL database named {MYSQL_DATABASE}.

RELEVANT DATABASE SCHEMA FOR THIS QUESTION:
{schema_context}

User Question: "{user_question}"

CRITICAL INSTRUCTIONS:
1. Generate ONLY a valid MySQL SELECT or SHOW query
2. Use ONLY the tables and columns shown in the schema above
3. Return ONLY the SQL query with NO explanations or markdown
4. Do NOT invent column names - use only what's in the schema

SPECIAL CASES:
- For "how many tables": SELECT COUNT(DISTINCT table_name) FROM information_schema.tables WHERE table_schema = '{MYSQL_DATABASE}'
- For "show/list tables": SELECT table_name FROM information_schema.tables WHERE table_schema = '{MYSQL_DATABASE}'
- For "describe table X": SELECT column_name, data_type FROM information_schema.columns WHERE table_schema = '{MYSQL_DATABASE}' AND table_name = 'X'

EXAMPLES:
User: "how many customers?"
SQL: SELECT COUNT(*) AS customer_count FROM customers

User: "list products"
SQL: SELECT * FROM products LIMIT 50

User: "how many tables in database?"
SQL: SELECT COUNT(DISTINCT table_name) AS table_count FROM information_schema.tables WHERE table_schema = '{MYSQL_DATABASE}'

Return ONLY the SQL query:
"""

def get_relevant_schema_context(user_question: str, top_k: int = 15) -> str:
    """Get relevant schema information based on user question using RAG"""
    try:
        # Embed the user question
        q_emb = embed_texts([user_question])[0]
        
        # Query schema collection for relevant schema info
        results = schema_coll.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["documents", "metadatas"]
        )
        
        if not results or not results.get('documents') or not results['documents'][0]:
            return f"Database: {MYSQL_DATABASE}"

        # Organize schema information by table
        tables_info = {}
        for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
            table = meta.get('table', 'unknown')
            column = meta.get('column', 'unknown')
            data_type = meta.get('data_type', 'unknown')
            
            if table not in tables_info:
                tables_info[table] = []
            tables_info[table].append(f"{column} ({data_type})")
        
        # Format schema context
        schema_lines = []
        for table, columns in sorted(tables_info.items()):
            schema_lines.append(f"Table: {table}")
            schema_lines.append(f"  Columns: {', '.join(columns)}")

        return "\n".join(schema_lines) if schema_lines else f"Database: {MYSQL_DATABASE}"

    except Exception as e:
        print(f"Warning: Could not get schema context: {e}")
        return f"Database: {MYSQL_DATABASE}"

def generate_sql_via_gemini(user_question: str) -> str:
    """Generate SQL query using Gemini API with RAG schema context"""
    # Get relevant schema information using RAG
    schema_context = get_relevant_schema_context(user_question, top_k=20)
    
    prompt = GEN_SQL_PROMPT_TEMPLATE.format(
        user_question=user_question,
        schema_context=schema_context,
        MYSQL_DATABASE=MYSQL_DATABASE,
    )
    
    try:
        response = client.chat.completions.create(
            model=GEMINI_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert MySQL database assistant. Generate only valid, complete SQL queries using ONLY the provided schema."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,  # Set to 0 for more deterministic results
            max_tokens=512,
        )
        
        msg = response.choices[0].message.content.strip()
        
        # Check for clarification request
        if msg.upper().startswith("CLARIFY"):
            return msg
        
        # Remove markdown code blocks
        if "```sql" in msg.lower():
            parts = msg.lower().split("```sql")
            if len(parts) > 1:
                msg = parts[1].split("```")[0].strip()
        elif "```" in msg:
            parts = msg.split("```")
            if len(parts) > 1:
                msg = parts[1].split("```")[0].strip()
        
        # Clean up the query
        lines = [line.strip() for line in msg.splitlines() if line.strip() and not line.strip().startswith('#')]
        
        # Find the actual SQL query
        sql_query = None
        for i, line in enumerate(lines):
            lower = line.lower()
            if lower.startswith(("select", "with", "show")):
                # Join all lines from this point to construct complete query
                sql_query = " ".join(lines[i:])
                break
        
        if not sql_query:
            # If no SELECT/SHOW found, try to use the entire cleaned message
            sql_query = " ".join(lines).strip()

        if not sql_query:
            raise ValueError("Model did not return a usable SQL query.")
        
        # Remove any trailing semicolons for now (we'll add them back if needed)
        sql_query = sql_query.rstrip(';').strip()
        
        # Validate the query has basic structure
        sql_lower = sql_query.lower()
        if sql_lower.startswith("select"):
            # Check if it has FROM clause (unless it's a system query or just a count)
            if "from" not in sql_lower:
                raise ValueError(f"Generated incomplete SQL query (missing FROM clause): {sql_query}")
        
        print(f"[DEBUG] Schema context used:\n{schema_context[:200]}...")
        return sql_query
        
    except Exception as e:
        print(f"Error generating SQL: {e}")
        raise

def execute_sql_and_fetch(sql: str, limit: int = 100) -> pd.DataFrame:
    """Execute SQL query and return results as DataFrame"""
    lowered = sql.strip().lower()
    
    # Allow ONLY SELECT and SHOW queries for safety
    if not (lowered.startswith("select") or lowered.startswith("show")):
        raise ValueError("Only SELECT and SHOW queries are allowed for safety")
    
    # Prevent malicious queries - check for dangerous keywords
    dangerous_keywords = ["drop", "delete", "insert", "update", "truncate", "alter", "create", "grant", "revoke"]
    sql_words = lowered.split()
    for keyword in dangerous_keywords:
        if keyword in sql_words:
            raise ValueError(f"Query contains forbidden keyword: {keyword}")
    
    engine = get_sqlalchemy_engine()
    safe_sql = sql.rstrip(';')
    
    # Add LIMIT if not present and it's a SELECT query (not information_schema)
    if lowered.startswith("select") and "limit" not in lowered and "information_schema" not in lowered:
        safe_sql = f"{safe_sql} LIMIT {limit}"
    
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(safe_sql), conn)
        return df
    except Exception as e:
        print(f"SQL execution error: {e}")
        raise

def result_to_nl_answer(df: pd.DataFrame, user_question: str) -> str:
    """Convert DataFrame results to natural language answer"""
    if df is None or df.shape[0] == 0:
        return "No results found for your query."
    
    n = df.shape[0]
    preview_rows = min(5, n)
    preview = df.head(preview_rows).to_dict(orient="records")
    
    lines = []
    for row in preview:
        line = " | ".join(f"{k}: {v}" for k, v in row.items())
        lines.append(line)
    
    result = f"Found {n} row{'s' if n != 1 else ''}.\n\n"
    if n <= preview_rows:
        result += "Results:\n" + "\n".join(lines)
    else:
        result += f"Showing first {preview_rows} results:\n" + "\n".join(lines)
        result += f"\n\n(+ {n - preview_rows} more rows)"
    
    return result

def ask(user_question: str, cache_threshold: float = 0.85) -> Dict:
    """Main function to answer user questions"""
    if not user_question.strip():
        return {"source": "error", "error": "Empty question provided"}
    
    # Check cache first
    cached, score = query_cache_find_similar(user_question, top_k=1, score_threshold=cache_threshold)
    if cached:
        print(f"[CACHE HIT] Similarity score: {score:.2f}")
        return {
            "source": "cache",
            "question": cached["question"],
            "sql": cached["sql"],
            "answer": cached["answer"]
        }
    
    print("[CACHE MISS] Generating new SQL query...")
    
    try:
        sql = generate_sql_via_gemini(user_question)
        
        # Check if clarification is needed
        if sql.startswith("CLARIFY:"):
            return {
                "source": "clarification",
                "message": sql.replace("CLARIFY:", "").strip()
            }
        
        print(f"Generated SQL: {sql}")
        
        # Execute query
        df = execute_sql_and_fetch(sql, limit=200)
        answer = result_to_nl_answer(df, user_question)
        
        # Cache the result
        cache_query_result(user_question, sql, answer)
        
        return {
            "source": "live",
            "sql": sql,
            "answer": answer,
            "rows": df.shape[0]
        }
        
    except Exception as e:
        return {
            "source": "error",
            "error": str(e),
            "sql": sql if 'sql' in locals() else "No SQL generated"
        }

def get_database_summary():
    """Get a summary of available tables in the database"""
    try:
        engine = get_sqlalchemy_engine()
        with engine.connect() as conn:
            # Get all tables
            tables_query = text("""
                SELECT table_name, table_rows 
                FROM information_schema.tables 
                WHERE table_schema = :schema
                ORDER BY table_name
            """)
            df = pd.read_sql(tables_query, conn, params={"schema": MYSQL_DATABASE})
            return df
    except Exception as e:
        print(f"Warning: Could not fetch database summary: {e}")
        return None

def main():
    """Main interactive loop"""
    global cache_coll
    print("=" * 60)
    print("RAG-based SQL Query System")
    print("=" * 60)
    
    try:
        print("\nInitializing schema collection...")
        populate_schema_collection(overwrite=False)
        
        # Clear old cache to prevent issues with cached bad queries
        print("Checking query cache...")
        try:
            cache_count = cache_coll.count()
            if cache_count > 0:
                print(f"Found {cache_count} cached queries.")
                clear_cache = input("Clear old cache? (y/n, default=n): ").strip().lower()
                if clear_cache == 'y':
                    chroma_client.delete_collection("query_cache")
                    cache_coll = chroma_client.create_collection(
                        name="query_cache",
                        metadata={"description": "Cached query results"}
                    )
                    print("✓ Cache cleared successfully")
        except Exception as e:
            print(f"Note: Could not check cache: {e}")
        
        # Show database summary
        print("\n" + "=" * 60)
        print("DATABASE SUMMARY")
        print("=" * 60)
        summary = get_database_summary()
        if summary is not None and not summary.empty:
            print(f"\nDatabase: {MYSQL_DATABASE}")
            print(f"Total tables: {len(summary)}\n")
            print(summary.to_string(index=False))
        print("=" * 60)
        
        print("\n✓ System ready!")
        print("\nYou can now ask questions about the database.")
        print("Examples:")
        print("  - How many tables in this database?")
        print("  - Show me all tables")
        print("  - How many customers?")
        print("  - List all products")
        print("\nType 'exit' or 'quit' to stop.\n")
        
        while True:
            try:
                q = input("\n🔍 Question> ").strip()
                
                if not q:
                    continue
                
                if q.lower() in ("exit", "quit", "q"):
                    print("\nGoodbye!")
                    break
                
                resp = ask(q)
                
                if resp.get("source") == "error":
                    print(f"\n❌ Error: {resp.get('error')}")
                    if resp.get("sql"):
                        print(f"SQL attempted: {resp.get('sql')}")
                
                elif resp.get("source") == "clarification":
                    print(f"\n❓ {resp.get('message')}")
                
                elif resp.get("source") == "cache":
                    print(f"\n💾 Answer (from cache):\n{resp.get('answer')}")
                    print(f"\n📝 SQL used:\n{resp.get('sql')}")
                
                else:
                    print(f"\n✓ Answer:\n{resp.get('answer')}")
                    print(f"\n📝 SQL executed:\n{resp.get('sql')}")
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Type 'exit' to quit.")
                continue
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
                import traceback
                traceback.print_exc()
                continue
                
    except Exception as e:
        print(f"\n❌ Fatal error during initialization: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()