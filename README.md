E-Commerce Customer Intelligence & Churn Prediction
│
├── app/
│   ├── __init__.py
│   ├── app.py
│   │
│   ├── templates/
│   │   ├── base.html
│   │   ├── index.html
│   │   ├── dashboard.html
│   │   ├── upload.html
│   │   ├── customers.html
│   │   ├── customer-detail.html
│   │   ├── segments.html
│   │   ├── churn.html
│   │   ├── recommendations.html
│   │   ├── analytics.html
│   │   ├── reports.html
│   │   ├── settings.html
│   │   ├── login.html
│   │   └── error.html
│   │
│   └── static/
│       ├── css/
│       │   └── style.css
│       ├── js/
│       │   ├── app.js
│       │   ├── dashboard.js
│       │   └── charts.js
│       ├── images/
│       └── icons/
│
├── data/
│   ├── raw/
│   │   ├── olist_customers_dataset.csv
│   │   ├── olist_geolocation_dataset.csv
│   │   ├── olist_order_items_dataset.csv
│   │   ├── olist_order_payments_dataset.csv
│   │   ├── olist_order_reviews_dataset.csv
│   │   ├── olist_orders_dataset.csv
│   │   ├── olist_products_dataset.csv
│   │   ├── olist_sellers_dataset.csv
│   │   └── product_category_name_translation.csv
│   │
│   ├── processed/
│   ├── modeling/
│   ├── templates/
│   ├── sample/
│   └── uploads/
│
├── notebooks/
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── audit.py
│   │   ├── loader.py
│   │   ├── validator.py
│   │   ├── cleaner.py
│   │   └── processor.py
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── customer_features.py
│   │   ├── rfm.py
│   │   ├── clv.py
│   │   └── churn_features.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── predict.py
│   │   └── explain.py
│   │
│   ├── analytics/
│   │   ├── __init__.py
│   │   ├── customer_analytics.py
│   │   ├── sales_analytics.py
│   │   ├── product_analytics.py
│   │   └── retention_analytics.py
│   │
│   ├── business/
│   │   ├── __init__.py
│   │   ├── recommendations.py
│   │   ├── customer_segments.py
│   │   └── retention_actions.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       ├── helpers.py
│       └── config.py
│
├── models/
│
├── reports/
│   ├── figures/
│   └── exports/
│
├── tests/
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_features.py
│   └── test_model.py
│
├── config/
│   └── app_config.py
│
├── docs/
│
├── requirements.txt
├── requirements-dev.txt
├── .gitignore
├── README.md
└── LICENSE