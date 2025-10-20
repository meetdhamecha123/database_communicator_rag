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
from datetime import datetime

# --- Load config ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", 3306))
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CSV_EXPORT_DIR = os.getenv("CSV_EXPORT_DIR", "./query_results")

if not GEMINI_API_KEY:
    raise RuntimeError("Please set GEMINI_API_KEY in your .env file")

# Create CSV export directory
os.makedirs(CSV_EXPORT_DIR, exist_ok=True)

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
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        print(f"ChromaDB initialized at: {CHROMA_PERSIST_DIR}")
        
    except Exception as e:
        print(f"Warning: Could not use PersistentClient: {e}")
        print("Falling back to ephemeral client (data will not persist)")
        chroma_client = chromadb.Client()
    
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

initialize_chroma()

def build_document_text_from_row(row: Dict) -> str:
    """Build searchable document text from schema row"""
    parts = [
        f"TABLE: {row.get('TABLE NAME', '')}",
        f"COLUMN: {row.get('COLUMN NAME', '')}",
        f"TYPE: {row.get('DATA TYPE', '')}",
        f"KEY: {row.get('COLUMN KEY', '')}",
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
            "column_key": str(r.get("COLUMN KEY", "")),
        })
        ids.append(f"schema_row_{i}_{r.get('TABLE NAME')}_{r.get('COLUMN NAME')}")
    
    embeddings = embed_texts(texts)
    
    if overwrite:
        print("Rebuilding schema collection...")
        try:
            chroma_client.delete_collection("schema")
        except Exception as e:
            print(f"Note: Could not delete collection: {e}")
        
        global schema_coll
        schema_coll = chroma_client.create_collection(name="schema")
    
    try:
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
    score_threshold: float = 0.88
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
    except Exception as e:
        print(f"Warning: Could not cache result: {e}")

# --- Enhanced Schema Context ---
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

def get_table_relationships() -> str:
    """Get foreign key relationships"""
    try:
        engine = get_sqlalchemy_engine()
        with engine.connect() as conn:
            fk_query = text("""
                SELECT 
                    TABLE_NAME,
                    COLUMN_NAME,
                    REFERENCED_TABLE_NAME,
                    REFERENCED_COLUMN_NAME
                FROM information_schema.KEY_COLUMN_USAGE
                WHERE TABLE_SCHEMA = :schema
                AND REFERENCED_TABLE_NAME IS NOT NULL
            """)
            df = pd.read_sql(fk_query, conn, params={"schema": MYSQL_DATABASE})
            
            if df.empty:
                return ""
            
            relationships = []
            for _, row in df.iterrows():
                relationships.append(
                    f"{row['TABLE_NAME']}.{row['COLUMN_NAME']} -> "
                    f"{row['REFERENCED_TABLE_NAME']}.{row['REFERENCED_COLUMN_NAME']}"
                )
            return "Foreign Key Relationships:\n" + "\n".join(relationships)
    except Exception as e:
        return ""

def get_relevant_schema_context(user_question: str, top_k: int = 30) -> str:
    """Get comprehensive schema context with better organization"""
    try:
        actual_tables = get_all_table_names()
        q_emb = embed_texts([user_question])[0]
        
        results = schema_coll.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["documents", "metadatas"]
        )
        
        if not results or not results.get('documents') or not results['documents'][0]:
            return f"Database: {MYSQL_DATABASE}\nTables: {', '.join(actual_tables)}"

        # Organize by table with primary keys highlighted
        tables_info = {}
        for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
            table = meta.get('table', 'unknown')
            column = meta.get('column', 'unknown')
            data_type = meta.get('data_type', 'unknown')
            col_key = meta.get('column_key', '')
            
            if table in actual_tables:
                if table not in tables_info:
                    tables_info[table] = {'columns': [], 'pk': []}
                
                col_desc = f"{column} ({data_type})"
                if col_key == 'PRI':
                    tables_info[table]['pk'].append(column)
                    col_desc += " [PRIMARY KEY]"
                elif col_key == 'MUL':
                    col_desc += " [FOREIGN KEY]"
                
                tables_info[table]['columns'].append(col_desc)
        
        # Build context
        context_parts = [
            f"=== DATABASE: {MYSQL_DATABASE} ===",
            f"Available Tables: {', '.join(actual_tables)}",
            "",
            "=== DETAILED SCHEMA ===",
        ]
        
        for table, info in sorted(tables_info.items()):
            context_parts.append(f"\nTable: {table}")
            if info['pk']:
                context_parts.append(f"  Primary Key: {', '.join(info['pk'])}")
            context_parts.append(f"  Columns: {', '.join(info['columns'][:15])}")
        
        # Add relationships
        relationships = get_table_relationships()
        if relationships:
            context_parts.append(f"\n{relationships}")
        
        return "\n".join(context_parts)

    except Exception as e:
        print(f"Warning: Error in schema context: {e}")
        actual_tables = get_all_table_names()
        return f"Database: {MYSQL_DATABASE}\nTables: {', '.join(actual_tables)}"

# --- SQL Generation ---
GEN_SQL_PROMPT_TEMPLATE = """You are an expert MySQL database assistant for the {MYSQL_DATABASE} database.

{schema_context}

USER QUESTION: "{user_question}"

INSTRUCTIONS:
1. Analyze the question carefully and use the schema above
2. Generate ONE valid MySQL query using ONLY the exact table/column names shown
3. Return ONLY the SQL query - no explanations, no markdown
4. Use proper JOINs when querying related tables
5. Keep queries efficient and simple

CRITICAL RULES:
- Use exact table names (e.g., if schema shows "customer", use "customer" NOT "customers")
- Never invent table or column names
- For "most/maximum rows" questions: use UNION ALL pattern
- For aggregations: use proper GROUP BY clauses

Return only the SQL query:"""

def clean_sql_response(raw_response: str) -> str:
    """Clean SQL response from LLM"""
    msg = raw_response.strip()
    
    if "```sql" in msg.lower():
        parts = msg.lower().split("```sql")
        if len(parts) > 1:
            msg = parts[1].split("```")[0].strip()
    elif "```" in msg:
        parts = msg.split("```")
        if len(parts) > 1:
            msg = parts[1].split("```")[0].strip()
    
    lines = []
    for line in msg.splitlines():
        line = line.strip()
        if not line or line.startswith('--') or line.startswith('#'):
            continue
        if '--' in line:
            line = line.split('--')[0].strip()
        if line:
            lines.append(line)
    
    sql_query = " ".join(lines).strip().rstrip(';').strip()
    return sql_query

def validate_sql_query(sql: str) -> tuple[bool, str]:
    """Validate SQL query"""
    if not sql:
        return False, "Empty query"
    
    sql_lower = sql.lower().strip()
    
    if not (sql_lower.startswith("select") or sql_lower.startswith("show")):
        return False, "Must start with SELECT or SHOW"
    
    dangerous = ["drop", "delete", "insert", "update", "truncate", "alter", "create"]
    if any(kw in sql_lower.split() for kw in dangerous):
        return False, "Contains forbidden keyword"
    
    if sql_lower.startswith("select"):
        if sql.count('(') != sql.count(')'):
            return False, "Unbalanced parentheses"
    
    return True, "Valid"

def generate_sql_via_gemini(user_question: str, max_retries: int = 2) -> str:
    """Generate SQL with enhanced schema understanding"""
    
    # Special handling for largest table queries
    question_lower = user_question.lower()
    if any(p in question_lower for p in ['maximum row', 'most row', 'largest table', 'biggest table']):
        print("[INFO] Detected 'largest table' query")
        try:
            tables = get_all_table_names()
            if tables:
                union_parts = [f"SELECT '{t}' AS table_name, COUNT(*) AS row_count FROM {t}" for t in tables]
                return " UNION ALL ".join(union_parts) + " ORDER BY row_count DESC LIMIT 1"
        except Exception as e:
            print(f"[WARNING] Fallback to LLM: {e}")
    
    schema_context = get_relevant_schema_context(user_question, top_k=30)
    
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
                    {"role": "system", "content": "You are a MySQL expert. Use ONLY exact table/column names from the schema. Return ONLY valid SQL queries."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=1200,
            )
            
            raw_response = response.choices[0].message.content.strip()
            sql_query = clean_sql_response(raw_response)
            
            if not sql_query:
                if attempt < max_retries - 1:
                    continue
                raise ValueError("Empty SQL query")
            
            is_valid, error_msg = validate_sql_query(sql_query)
            if not is_valid:
                if attempt < max_retries - 1:
                    prompt += f"\n\nERROR: {error_msg}. Fix and return only the corrected SQL."
                    continue
                raise ValueError(f"Invalid SQL: {error_msg}")
            
            print(f"[SQL Generated] {sql_query[:80]}...")
            return sql_query
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            raise

def execute_sql_and_fetch(sql: str, limit: int = 100) -> pd.DataFrame:
    """Execute SQL and return full results"""
    is_valid, error_msg = validate_sql_query(sql)
    if not is_valid:
        raise ValueError(f"Invalid SQL: {error_msg}")
    
    engine = get_sqlalchemy_engine()
    safe_sql = sql.rstrip(';')
    
    # Don't add LIMIT for counting queries or information_schema
    sql_lower = safe_sql.lower()
    needs_limit = (
        sql_lower.startswith("select") and 
        "limit" not in sql_lower and 
        "information_schema" not in sql_lower and
        "count(*)" not in sql_lower
    )
    
    try:
        with engine.connect() as conn:
            # Execute without LIMIT to get full results
            df = pd.read_sql(text(safe_sql), conn)
        return df
    except Exception as e:
        print(f"SQL error: {e}")
        raise

# --- Natural Language Answer Generation ---
def generate_nl_answer(user_question: str, sql: str, df: pd.DataFrame, csv_file: Optional[str] = None) -> str:
    """Generate natural language answer using Gemini"""
    
    if df is None or df.shape[0] == 0:
        return "No results found for your query."
    
    # Prepare data summary
    row_count = df.shape[0]
    col_count = df.shape[1]
    
    # Get sample data (first 10 rows)
    sample_data = df.head(10).to_dict(orient='records')
    
    # Create prompt for NL generation
    prompt = f"""Based on the following database query results, provide a clear, natural language answer to the user's question.

USER QUESTION: "{user_question}"

SQL QUERY EXECUTED: {sql}

RESULTS SUMMARY:
- Total rows: {row_count}
- Columns: {', '.join(df.columns.tolist())}

SAMPLE DATA (first 10 rows):
{json.dumps(sample_data, indent=2, default=str)}

INSTRUCTIONS:
1. Provide a direct answer to the user's question
2. Use natural, conversational language
3. Highlight key findings and insights
4. Include specific numbers and values from the data
5. If there are many rows, summarize the key patterns
6. Keep it concise but informative

Your natural language answer:"""

    try:
        response = client.chat.completions.create(
            model=GEMINI_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful data analyst. Provide clear, natural language answers based on database query results."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=500,
        )
        
        nl_answer = response.choices[0].message.content.strip()
        
        # Add CSV file info if applicable
        if csv_file:
            nl_answer += f"\n\n📄 Complete results ({row_count} rows) saved to: {csv_file}"
        
        return nl_answer
        
    except Exception as e:
        print(f"Warning: Could not generate NL answer: {e}")
        # Fallback to basic summary
        return f"Found {row_count} rows with {col_count} columns. " + \
               f"Sample: {df.head(3).to_string(index=False)}"

# --- CSV Export ---
def save_to_csv(df: pd.DataFrame, question: str) -> str:
    """Save DataFrame to CSV file"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Clean filename
        clean_q = "".join(c if c.isalnum() or c in (' ', '_') else '_' for c in question)
        clean_q = clean_q[:50].strip().replace(' ', '_')
        
        filename = f"query_{timestamp}_{clean_q}.csv"
        filepath = os.path.join(CSV_EXPORT_DIR, filename)
        
        df.to_csv(filepath, index=False, encoding='utf-8')
        print(f"[CSV Export] Saved {len(df)} rows to: {filepath}")
        return filepath
        
    except Exception as e:
        print(f"Warning: Could not save CSV: {e}")
        return None

# --- Main Query Function ---
def ask(user_question: str, cache_threshold: float = 0.88, csv_threshold: int = 50) -> Dict:
    """Main function with NL answers and CSV export"""
    if not user_question.strip():
        return {"source": "error", "error": "Empty question"}
    
    # Check cache
    cached, score = query_cache_find_similar(user_question, score_threshold=cache_threshold)
    if cached:
        print(f"[CACHE HIT] Score: {score:.2f}")
        return {
            "source": "cache",
            "question": cached["question"],
            "sql": cached["sql"],
            "answer": cached["answer"]
        }
    
    print("[CACHE MISS] Generating new query...")
    
    sql = None
    try:
        sql = generate_sql_via_gemini(user_question)
        print(f"[Executing SQL] {sql}")
        
        # Execute query - get ALL results
        df = execute_sql_and_fetch(sql)
        row_count = df.shape[0]
        
        # Save to CSV if results are large
        csv_file = None
        if row_count > csv_threshold:
            csv_file = save_to_csv(df, user_question)
        
        # Generate natural language answer
        nl_answer = generate_nl_answer(user_question, sql, df, csv_file)
        
        # Cache the result
        cache_query_result(user_question, sql, nl_answer)
        
        return {
            "source": "live",
            "sql": sql,
            "answer": nl_answer,
            "rows": row_count,
            "csv_file": csv_file
        }
        
    except Exception as e:
        return {
            "source": "error",
            "error": str(e),
            "sql": sql if sql else "No SQL generated"
        }

def get_database_summary():
    """Get database summary"""
    try:
        engine = get_sqlalchemy_engine()
        with engine.connect() as conn:
            tables_query = text("""
                SELECT table_name, table_rows 
                FROM information_schema.tables 
                WHERE table_schema = :schema
                ORDER BY table_rows DESC
            """)
            df = pd.read_sql(tables_query, conn, params={"schema": MYSQL_DATABASE})
            return df
    except Exception as e:
        print(f"Warning: Could not fetch summary: {e}")
        return None

def main():
    """Main loop"""
    global cache_coll
    print("=" * 70)
    print("🚀 Enhanced RAG-based SQL Query System with NL Answers")
    print("=" * 70)
    
    try:
        print("\n[1/3] Initializing schema collection...")
        populate_schema_collection(overwrite=False)
        
        print("\n[2/3] Checking cache...")
        try:
            cache_count = cache_coll.count()
            if cache_count > 0:
                print(f"Found {cache_count} cached queries.")
                clear = input("Clear cache? (y/n, default=n): ").strip().lower()
                if clear == 'y':
                    chroma_client.delete_collection("query_cache")
                    cache_coll = chroma_client.create_collection(name="query_cache")
                    print("✓ Cache cleared")
        except Exception as e:
            print(f"Note: {e}")
        
        print("\n[3/3] Loading database summary...")
        summary = get_database_summary()
        if summary is not None and not summary.empty:
            print(f"\n{'='*70}")
            print(f"DATABASE: {MYSQL_DATABASE}")
            print(f"Total Tables: {len(summary)}")
            print(f"{'='*70}")
            print(summary.to_string(index=False, max_rows=10))
            if len(summary) > 10:
                print(f"... and {len(summary) - 10} more tables")
            print(f"{'='*70}")
        
        print("\n✅ System Ready!")
        print(f"\n📁 Large results (>50 rows) will be saved to: {CSV_EXPORT_DIR}")
        print("\n💡 Examples:")
        print("  • How many customers do we have?")
        print("  • Which table has the most records?")
        print("  • Show me top 10 products by price")
        print("  • List all employees with their departments")
        print("\nType 'exit' or 'quit' to stop.\n")
        
        while True:
            try:
                q = input("\n🔍 Question> ").strip()
                
                if not q:
                    continue
                
                if q.lower() in ("exit", "quit", "q"):
                    print("\n👋 Goodbye!")
                    break
                
                print()  # Add spacing
                resp = ask(q)
                
                if resp.get("source") == "error":
                    print(f"❌ Error: {resp.get('error')}")
                    if resp.get("sql"):
                        print(f"\n📝 SQL attempted:\n{resp.get('sql')}")
                
                elif resp.get("source") == "cache":
                    print(f"💾 Answer (cached):\n{resp.get('answer')}")
                    print(f"\n📝 SQL:\n{resp.get('sql')}")
                
                else:
                    print(f"✅ Answer:\n{resp.get('answer')}")
                    print(f"\n📝 SQL ({resp.get('rows')} rows):\n{resp.get('sql')}")
                
            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted. Type 'exit' to quit.")
                continue
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()
                
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()