import pandas as pd
from sqlalchemy import create_engine

# 1. DATABASE CONNECTION STRINGS
# Point this to your local SQLite file
SQLITE_CONN = "sqlite:////home/likith/mlops/MLOPS/Data/raw/olist.sqlite"
# Point this to your target PostgreSQL database
POSTGRES_CONN = "postgresql://postgres:postgres@localhost:5432/olist_destination"

def run_etl():
    print("🚀 Initializing E-Commerce ETL Pipeline...")
    
    sqlite_engine = create_engine(SQLITE_CONN)
    pg_engine = create_engine(POSTGRES_CONN)
    
    # 2. DEFINING THE STRICT ELT ORDER (Prevents Foreign Key Failures)
    ordered_tables = [
        "product_category_name_translation",
        "geolocation",
        "customers",
        "leads_qualified",
        "products",
        "sellers",
        "orders",
        "leads_closed",
        "order_items",
        "order_payments",
        "order_reviews"
    ]
    
    for table in ordered_tables:
        print(f"\n📥 Extracting data from table: '{table}'...")
        try:
            # --- EXTRACT ---
            df = pd.read_sql_table(table, con=sqlite_engine)
            
            # --- TRANSFORM ---
            # SQLite stores dates as strings. PostgreSQL strictly expects timestamp objects.
            # We automatically find and convert any date/timestamp columns.
            date_cols = [col for col in df.columns if 'date' in col or 'timestamp' in col or 'approved_at' in col]
            for col in date_cols:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # Handle unique IDs that might have formatting issues
            if 'geolocation_zip_code_prefix' in df.columns:
                df['geolocation_zip_code_prefix'] = df['geolocation_zip_code_prefix'].astype(str)
                
            print(f"🔄 Transformed {len(date_cols)} datetime columns for '{table}'.")
            
            # --- LOAD ---
            print(f"📤 Loading '{table}' ({len(df)} rows) into PostgreSQL...")
            df.to_sql(
                name=table,
                con=pg_engine,
                if_exists='append',  # 'append' preserves pre-defined primary/foreign keys
                index=False,
                method='multi',      # Speeds up processing by grouping row inserts
                chunksize=10000      # Prevents memory overload on huge tables like order_items
            )
            print(f"✅ Table '{table}' successfully migrated.")
            
        except Exception as e:
            print(f"❌ Error occurred while migrating '{table}': {str(e)}")
            print("Stopping execution to maintain data integrity.")
            break

    print("\n🏁 ETL Pipeline Execution Finished!")

if __name__ == "__main__":
    run_etl()
