#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis (EDA) on E-Commerce Dataset
# 
# This notebook connects to the PostgreSQL database `olist_destination` to perform exploratory data analysis and visualize key metrics of our e-commerce data.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, text

# Set up seaborn aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 12

# Connection String
POSTGRES_CONN = "postgresql://postgres:postgres@localhost:5432/olist_destination"
engine = create_engine(POSTGRES_CONN)


# ## 1. Monthly Revenue Trend
# 
# Let's see how the monthly revenue changes over time.

# In[3]:


revenue_query = """
    SELECT 
        DATE_TRUNC('month', o.order_purchase_timestamp) AS month,
        SUM(p.payment_value) AS revenue
    FROM orders o
    JOIN order_payments p ON o.order_id = p.order_id
    WHERE o.order_status = 'delivered'
    GROUP BY month
    ORDER BY month;
"""

with engine.connect() as conn:
    df_revenue = pd.read_sql_query(text(revenue_query), conn)

# Clean up month format
df_revenue['month'] = pd.to_datetime(df_revenue['month'])
df_revenue = df_revenue[df_revenue['month'] < '2018-09-01']  # filter incomplete last month if any

plt.figure(figsize=(14, 6))
sns.lineplot(data=df_revenue, x='month', y='revenue', marker='o', color='royalblue', linewidth=2.5)
plt.title('Monthly Sales Revenue Trend', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Revenue ($)', fontsize=12)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ## 2. Top 10 Product Categories by Revenue
# 
# Let's visualize the top 10 categories generating the highest sales revenue.

# In[4]:


top_cats_query = """
    SELECT 
        t.product_category_name_english AS category,
        SUM(i.price) AS revenue
    FROM order_items i
    JOIN products p ON i.product_id = p.product_id
    JOIN product_category_name_translation t ON p.product_category_name = t.product_category_name
    GROUP BY t.product_category_name_english
    ORDER BY revenue DESC
    LIMIT 10;
"""

with engine.connect() as conn:
    df_top_cats = pd.read_sql_query(text(top_cats_query), conn)

sns.barplot(data=df_top_cats, x='revenue', y='category', palette='viridis', hue='category', legend=False)
plt.title('Top 10 Product Categories by Revenue', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Revenue ($)', fontsize=12)
plt.ylabel('Product Category', fontsize=12)
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
plt.tight_layout()
plt.show()


# ## 3. Order Payment Types
# 
# Let's see how customers prefer to pay for their orders.

# In[5]:


payment_query = """
    SELECT payment_type, COUNT(*) AS count
    FROM order_payments
    GROUP BY payment_type
    ORDER BY count DESC;
"""

with engine.connect() as conn:
    df_payment = pd.read_sql_query(text(payment_query), conn)

# Exclude any unknown types if necessary
df_payment = df_payment[df_payment['payment_type'] != 'not_defined']

plt.figure(figsize=(8, 8))
plt.pie(
    df_payment['count'], 
    labels=df_payment['payment_type'].str.replace('_', ' ').str.title(), 
    autopct='%1.1f%%', 
    colors=sns.color_palette('pastel'),
    startangle=140,
    textprops={'fontsize': 13}
)
plt.title('Distribution of Payment Types Used', fontsize=16, fontweight='bold', pad=15)
plt.tight_layout()
plt.show()


# ## 4. Delivery Days Distribution
# 
# Let's analyze the distribution of actual delivery times in days.

# In[6]:


delivery_query = """
    SELECT 
        EXTRACT(EPOCH FROM (order_delivered_customer_date - order_purchase_timestamp))/86400 AS delivery_days
    FROM orders
    WHERE order_status = 'delivered' 
      AND order_delivered_customer_date IS NOT NULL;
"""

with engine.connect() as conn:
    df_delivery = pd.read_sql_query(text(delivery_query), conn)

# Cap at 45 days for better visualization of the bulk distribution
df_filtered = df_delivery[df_delivery['delivery_days'] <= 45]

sns.histplot(df_filtered['delivery_days'], bins=30, kde=True, color='teal')
plt.axvline(df_delivery['delivery_days'].median(), color='red', linestyle='--', linewidth=2, label=f"Median: {df_delivery['delivery_days'].median():.1f} days")
plt.title('Distribution of Delivery Times (Days)', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Delivery Time (Days)', fontsize=12)
plt.ylabel('Number of Orders', fontsize=12)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

