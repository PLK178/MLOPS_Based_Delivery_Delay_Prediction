import os
import pandas as pd
from sqlalchemy import create_engine, text

# Connection strings
POSTGRES_CONN = "postgresql://postgres:postgres@localhost:5432/olist_destination"
OUTPUT_CSV_DIR = "/home/likith/mlops/MLOPS/Data/processed"
OUTPUT_CSV_PATH = os.path.join(OUTPUT_CSV_DIR, "final_dataset.csv")

import csv
from io import StringIO

def psql_insert_copy(table, conn, keys, data_iter):
    # gets a DBAPI connection from connection pool
    dbapi_conn = conn.connection
    with dbapi_conn.cursor() as cur:
        s_buf = StringIO()
        writer = csv.writer(s_buf)
        writer.writerows(data_iter)
        s_buf.seek(0)

        columns = ', '.join([f'"{k}"' for k in keys])
        if table.schema:
            table_name = f'"{table.schema}"."{table.name}"'
        else:
            table_name = f'"{table.name}"'

        sql = f"COPY {table_name} ({columns}) FROM STDIN WITH CSV NULL ''"
        cur.copy_expert(sql=sql, file=s_buf)

def perform_feature_engineering():
    print("🚀 Connecting to PostgreSQL database...")
    engine = create_engine(POSTGRES_CONN)
    
    query = """
        SELECT 
            o.order_id,
            o.customer_id,
            o.order_purchase_timestamp,
            o.order_approved_at,
            o.order_delivered_carrier_date,
            o.order_delivered_customer_date,
            o.order_estimated_delivery_date,
            i.product_id,
            i.seller_id,
            i.price,
            i.freight_value,
            p.product_category_name,
            p.product_weight_g,
            p.product_length_cm,
            p.product_height_cm,
            p.product_width_cm,
            c.customer_state,
            s.seller_state
        FROM orders o
        JOIN order_items i ON o.order_id = i.order_id
        JOIN products p ON i.product_id = p.product_id
        JOIN customers c ON o.customer_id = c.customer_id
        JOIN sellers s ON i.seller_id = s.seller_id
        WHERE o.order_status = 'delivered'
          AND o.order_delivered_customer_date IS NOT NULL;
    """
    
    print("📥 Loading raw dataset from database...")
    with engine.connect() as conn:
        df = pd.read_sql_query(text(query), conn)
    
    print(f"📊 Loaded {len(df)} records. Starting feature engineering...")
    
    # 1. Date/Time Features
    # Convert date columns to datetime objects
    date_cols = [
        'order_purchase_timestamp', 
        'order_approved_at', 
        'order_delivered_carrier_date', 
        'order_delivered_customer_date', 
        'order_estimated_delivery_date'
    ]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col])
        
    # Calculate time-based durations (in days)
    df['delivery_time_days'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.total_seconds() / 86400
    df['estimated_delivery_time_days'] = (df['order_estimated_delivery_date'] - df['order_purchase_timestamp']).dt.total_seconds() / 86400
    df['carrier_delivery_time_days'] = (df['order_delivered_carrier_date'] - df['order_purchase_timestamp']).dt.total_seconds() / 86400
    
    # Target variables for prediction
    df['delivery_delta_days'] = (df['order_delivered_customer_date'] - df['order_estimated_delivery_date']).dt.total_seconds() / 86400
    df['is_delayed'] = (df['delivery_delta_days'] > 0).astype(int)
    
    # Extract temporal components
    df['purchase_month'] = df['order_purchase_timestamp'].dt.month
    df['purchase_day_of_week'] = df['order_purchase_timestamp'].dt.dayofweek
    df['purchase_hour'] = df['order_purchase_timestamp'].dt.hour
    
    # 2. Product Physical Dimension Features
    # Fill missing dimensions with median values
    for col in ['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']:
        df[col] = df[col].fillna(df[col].median())
        
    # Volume calculation
    df['product_volume_cm3'] = df['product_length_cm'] * df['product_height_cm'] * df['product_width_cm']
    
    # 3. Financial/Freight Ratios
    df['freight_ratio'] = df['freight_value'] / (df['price'] + df['freight_value'])
    
    # 4. Geospatial Features
    # Binary feature checking if seller and customer are in the same state
    df['is_same_state'] = (df['customer_state'] == df['seller_state']).astype(int)
    
    # Drop raw timestamp and raw intermediate ID columns not needed for basic ML models
    # We include both delivery_time_days and delivery_delta_days as regression targets
    cols_to_keep = [
        'order_id', 'product_id', 'seller_id', 'price', 'freight_value', 
        'product_category_name', 'product_weight_g', 'product_volume_cm3',
        'is_same_state', 'purchase_month', 'purchase_day_of_week', 'purchase_hour',
        'estimated_delivery_time_days', 'delivery_time_days', 'delivery_delta_days', 'is_delayed'
    ]
    df_engineered = df[cols_to_keep]
    
    # 5. Save engineered features directly to CSV
    print(f"💾 Saving engineered features as CSV file to: {OUTPUT_CSV_PATH}")
    os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)
    df_engineered.to_csv(OUTPUT_CSV_PATH, index=False)

    
    print("✅ Feature Engineering Completed successfully!")

if __name__ == "__main__":
    perform_feature_engineering()
