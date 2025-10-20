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
GEN_SQL_PROMPT_TEMPLATE = """You are an expert SQL assistant for a MySQL database named {MYSQL_DATABASE}.

AVAILABLE TABLES AND SCHEMA:
{schema_context}

User Question: "{user_question}"

CRITICAL RULES:
1. Use ONLY the EXACT table names listed above - do NOT modify or pluralize them
2. Generate ONE complete, valid MySQL SELECT or SHOW query
3. Return ONLY the SQL query - NO explanations, NO markdown, NO comments
4. Keep queries SIMPLE - avoid complex nested queries
5. For questions about "which table has most/maximum rows", generate a UNION query for ALL tables listed above

CORRECT TABLE NAME USAGE:
- If you see "customer" in the schema, use "customer" NOT "customers"
- If you see "employee" in the schema, use "employee" NOT "employees"
- NEVER invent or guess table names - only use what's explicitly listed

QUERY PATTERNS:
- Count records: SELECT COUNT(*) FROM exact_table_name
- Find largest table: Generate UNION ALL of all available tables, then ORDER BY count DESC LIMIT 1
  Example: SELECT 'table1' AS table_name, COUNT(*) AS count FROM table1 UNION ALL SELECT 'table2', COUNT(*) FROM table2 ORDER BY count DESC LIMIT 1

EXAMPLES:
Q: "how many customers?"
Schema shows: customer
A: SELECT COUNT(*) AS count FROM customer

Q: "which table has most records?"
Schema shows: customer, employee, invoice
A: SELECT 'customer' AS table_name, COUNT(*) AS count FROM customer UNION ALL SELECT 'employee', COUNT(*) FROM employee UNION ALL SELECT 'invoice', COUNT(*) FROM invoice ORDER BY count DESC LIMIT 1

Q: "list all tables"
A: SELECT table_name FROM information_schema.tables WHERE table_schema = '{MYSQL_DATABASE}'

Return ONLY the SQL query (no explanations):
"""

def get_all_table_names() -> List[str]:
    """Get actual table names from database"""
    try:
        engine = get_sqlalchemy_engine()
        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = :schema ORDER BY table_name"
            ), {"schema": MYSQL_DATABASE})
            return [row[0] for row in result]
    except Exception as e:
        print(f"Warning: Could not fetch table names: {e}")
        return []

def get_relevant_schema_context(user_question: str, top_k: int = 15) -> str:
    """Get relevant schema information based on user question using RAG"""
    try:
        # First, get ALL actual table names from database
        actual_tables = get_all_table_names()
        
        # Embed the user question
        q_emb = embed_texts([user_question])[0]
        
        # Query schema collection for relevant schema info
        results = schema_coll.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["documents", "metadatas"]
        )
        
        if not results or not results.get('documents') or not results['documents'][0]:
            return f"Database: {MYSQL_DATABASE}\nAvailable tables: {', '.join(actual_tables)}"

        # Organize schema information by table
        tables_info = {}
        for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
            table = meta.get('table', 'unknown')
            column = meta.get('column', 'unknown')
            data_type = meta.get('data_type', 'unknown')
            
            # Only include if table actually exists
            if table in actual_tables:
                if table not in tables_info:
                    tables_info[table] = []
                tables_info[table].append(f"{column} ({data_type})")
        
        # Format schema context with actual table list
        schema_lines = [f"EXACT TABLE NAMES (use these exactly): {', '.join(actual_tables)}", ""]
        
        for table, columns in sorted(tables_info.items()):
            schema_lines.append(f"Table: {table}")
            schema_lines.append(f"  Columns: {', '.join(columns[:10])}")  # Limit columns to prevent overflow
            schema_lines.append("")

        return "\n".join(schema_lines) if len(schema_lines) > 2 else f"Database: {MYSQL_DATABASE}\nAvailable tables: {', '.join(actual_tables)}"

    except Exception as e:
        print(f"Warning: Could not get schema context: {e}")
        actual_tables = get_all_table_names()
        return f"Database: {MYSQL_DATABASE}\nAvailable tables: {', '.join(actual_tables)}"

def clean_sql_response(raw_response: str) -> str:
    """Clean and validate SQL response from LLM"""
    msg = raw_response.strip()
    
    # Remove markdown code blocks
    if "```sql" in msg.lower():
        parts = msg.lower().split("```sql")
        if len(parts) > 1:
            msg = parts[1].split("```")[0].strip()
    elif "```" in msg:
        parts = msg.split("```")
        if len(parts) > 1:
            msg = parts[1].split("```")[0].strip()
    
    # Remove comments and clean lines
    lines = []
    for line in msg.splitlines():
        line = line.strip()
        # Skip empty lines and comments
        if not line or line.startswith('--') or line.startswith('#'):
            continue
        # Remove inline comments
        if '--' in line:
            line = line.split('--')[0].strip()
        if line:
            lines.append(line)
    
    # Join lines into single query
    sql_query = " ".join(lines).strip()
    
    # Remove trailing semicolon
    sql_query = sql_query.rstrip(';').strip()
    
    return sql_query

def validate_sql_query(sql: str) -> tuple[bool, str]:
    """Validate SQL query structure and safety"""
    if not sql:
        return False, "Empty SQL query"
    
    sql_lower = sql.lower().strip()
    
    # Must start with SELECT or SHOW
    if not (sql_lower.startswith("select") or sql_lower.startswith("show")):
        return False, "Query must start with SELECT or SHOW"
    
    # Check for dangerous keywords
    dangerous = ["drop", "delete", "insert", "update", "truncate", "alter", "create", "grant", "revoke"]
    sql_words = sql_lower.split()
    for keyword in dangerous:
        if keyword in sql_words:
            return False, f"Query contains forbidden keyword: {keyword}"
    
    # Basic structure validation for SELECT queries
    if sql_lower.startswith("select"):
        # Check for balanced parentheses
        if sql.count('(') != sql.count(')'):
            return False, "Unbalanced parentheses in query"
        
        # For non-information_schema queries, should have FROM clause
        if "information_schema" not in sql_lower:
            if "from" not in sql_lower and "dual" not in sql_lower:
                return False, "SELECT query missing FROM clause"
    
    return True, "Valid"

def generate_sql_via_gemini(user_question: str, max_retries: int = 2) -> str:
    """Generate SQL query using Gemini API with validation and retry logic"""
    
    # Special handling for "maximum rows" type questions
    question_lower = user_question.lower()
    if any(phrase in question_lower for phrase in ['maximum row', 'most row', 'largest table', 'biggest table', 'which table has max']):
        print("[INFO] Detected 'largest table' query - using optimized approach")
        try:
            # Get all actual table names
            tables = get_all_table_names()
            if tables:
                # Build UNION query with actual table names
                union_parts = [f"SELECT '{table}' AS table_name, COUNT(*) AS row_count FROM {table}" for table in tables]
                sql_query = " UNION ALL ".join(union_parts) + " ORDER BY row_count DESC LIMIT 1"
                print(f"[DEBUG] Generated optimized SQL for largest table query")
                return sql_query
        except Exception as e:
            print(f"[WARNING] Could not generate optimized query: {e}, falling back to LLM")
    
    # Get relevant schema information using RAG
    schema_context = get_relevant_schema_context(user_question, top_k=25)
    
    prompt = GEN_SQL_PROMPT_TEMPLATE.format(
        user_question=user_question,
        schema_context=schema_context,
        MYSQL_DATABASE=MYSQL_DATABASE,
    )
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=GEMINI_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert MySQL assistant. Use ONLY the exact table names provided in the schema. Generate ONLY valid, complete, SIMPLE SQL queries. Return ONLY the SQL with no explanations."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=1000,  # Increased for longer UNION queries
            )
            
            raw_response = response.choices[0].message.content.strip()
            
            # Check for clarification request
            if raw_response.upper().startswith("CLARIFY"):
                return raw_response
            
            # Clean the response
            sql_query = clean_sql_response(raw_response)
            
            if not sql_query:
                if attempt < max_retries - 1:
                    print(f"[Attempt {attempt + 1}] Empty query, retrying...")
                    continue
                raise ValueError("Model returned empty SQL query")
            
            # Validate the query
            is_valid, error_msg = validate_sql_query(sql_query)
            
            if not is_valid:
                if attempt < max_retries - 1:
                    print(f"[Attempt {attempt + 1}] Invalid query: {error_msg}, retrying...")
                    # Add error context to next attempt
                    prompt += f"\n\nPREVIOUS ERROR: {error_msg}\nGenerate a simpler, valid query using ONLY the exact table names from the schema."
                    continue
                raise ValueError(f"Generated invalid SQL: {error_msg}")
            
            print(f"[DEBUG] Generated valid SQL: {sql_query[:100]}...")
            return sql_query
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"[Attempt {attempt + 1}] Error: {e}, retrying...")
                time.sleep(1)
                continue
            print(f"Error generating SQL after {max_retries} attempts: {e}")
            raise

def execute_sql_and_fetch(sql: str, limit: int = 100) -> pd.DataFrame:
    """Execute SQL query and return results as DataFrame"""
    # Final safety check
    is_valid, error_msg = validate_sql_query(sql)
    if not is_valid:
        raise ValueError(f"Cannot execute invalid SQL: {error_msg}")
    
    engine = get_sqlalchemy_engine()
    safe_sql = sql.rstrip(';')
    
    # Add LIMIT if not present and it's a SELECT query (not information_schema)
    sql_lower = safe_sql.lower()
    if sql_lower.startswith("select") and "limit" not in sql_lower and "information_schema" not in sql_lower:
        safe_sql = f"{safe_sql} LIMIT {limit}"
    
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(safe_sql), conn)
        return df
    except Exception as e:
        print(f"SQL execution error: {e}")
        print(f"Query was: {safe_sql}")
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
    
    sql = None
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
            "sql": sql if sql else "No SQL generated"
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