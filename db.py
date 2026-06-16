import os, psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from dotenv import load_dotenv

load_dotenv()
DB_URL = os.environ["DATABASE_URL"]

def init_db():
    with psycopg.connect(DB_URL) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.execute("""CREATE TABLE IF NOT EXISTS chunks (
            id SERIAL PRIMARY KEY,
            drug TEXT,
            section TEXT,
            content TEXT,
            embedding VECTOR(1536)   -- must match the embedding model's size
        );""")
    print("RAG DB ready.")

def add_fts():
    with psycopg.connect(DB_URL) as conn:
        conn.execute("""ALTER TABLE chunks ADD COLUMN IF NOT EXISTS content_tsv tsvector
            GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;""")
        conn.execute("CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING GIN (content_tsv);")
    print("Full-text search ready.")

def add_chunks(rows):                # rows: list of (drug, section, content, embedding)
    with psycopg.connect(DB_URL) as conn:
        register_vector(conn)        # lets psycopg send/receive vectors
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (drug, section, content, embedding) VALUES (%s,%s,%s,%s)",
                rows)

def search_chunks(query_embedding, k=5):
    with psycopg.connect(DB_URL) as conn:
        register_vector(conn)
        return conn.execute(
            "SELECT drug, section, content FROM chunks ORDER BY embedding <=> %s LIMIT %s",
            (Vector(query_embedding), k)).fetchall()

def vector_search(query_embedding, n=20):
    with psycopg.connect(DB_URL) as conn:
        register_vector(conn)
        return conn.execute(
            "SELECT id, drug, section, content FROM chunks ORDER BY embedding <=> %s LIMIT %s",
            (Vector(query_embedding), n)).fetchall()

def keyword_search(query, n=20):
    with psycopg.connect(DB_URL) as conn:
        return conn.execute(
            """SELECT id, drug, section, content
               FROM chunks
               WHERE content_tsv @@ plainto_tsquery('english', %s)
               ORDER BY ts_rank_cd(content_tsv, plainto_tsquery('english', %s)) DESC
               LIMIT %s""",
            (query, query, n)).fetchall()
