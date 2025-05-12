import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
import json
import uuid

class AdvancedSupplyChainDataGenerator:
    def __init__(self, 
                 num_products=50, 
                 num_suppliers=20, 
                 start_date=datetime(2021, 1, 1), 
                 end_date=datetime(2023, 12, 31)):
        """
        Comprehensive Supply Chain Data Generator
        
        :param num_products: Number of unique products to generate
        :param num_suppliers: Number of unique suppliers to generate
        :param start_date: Start date for historical data
        :param end_date: End date for historical data
        """
        self.num_products = num_products
        self.num_suppliers = num_suppliers
        self.start_date = start_date
        self.end_date = end_date
        
        # Data generation helpers
        self.product_categories = [
            'Electronics', 'Clothing', 'Home Goods', 
            'Food & Beverage', 'Automotive Parts', 
            'Industrial Equipment', 'Healthcare Supplies'
        ]
        
        self.regions = ['North America', 'Europe', 'Asia Pacific', 'Latin America', 'Middle East & Africa']
        self.sales_channels = ['Online', 'Retail', 'Wholesale', 'Direct', 'Distribution']
        
    def generate_sales_data(self):
        """
        Generate comprehensive sales dataset
        
        :return: Pandas DataFrame with sales data
        """
        sales_data = []
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        
        for _ in range(self.num_products):
            product_id = f'PROD_{str(uuid.uuid4())[:8]}'
            category = random.choice(self.product_categories)
            base_price = random.uniform(10, 500)
            
            for date in dates:
                # Simulate realistic sales patterns
                base_daily_sales = np.random.normal(50, 20)
                seasonal_factor = np.sin(date.month * np.pi / 6) * 10  # Seasonal variation
                day_of_week_factor = np.sin(date.dayofweek * np.pi / 3.5) * 5  # Day of week variation
                
                # Apply variations
                sales_quantity = max(0, int(base_daily_sales + seasonal_factor + day_of_week_factor))
                sales_value = sales_quantity * base_price
                
                sales_data.append({
                    'sales_id': str(uuid.uuid4()),
                    'product_id': product_id,
                    'date': date,
                    'sales_quantity': sales_quantity,
                    'sales_value': round(sales_value, 2),
                    'category': category,
                    'region': random.choice(self.regions),
                    'channel': random.choice(self.sales_channels),
                    'seasonality_flag': date.month in [11, 12, 1, 2],  # Winter season
                    'promotion_flag': random.random() < 0.1,  # 10% of sales during promotions
                    'unit_price': round(base_price, 2)
                })
        
        return pd.DataFrame(sales_data)
    
    def generate_suppliers_data(self):
        """
        Generate comprehensive suppliers dataset
        
        :return: Pandas DataFrame with suppliers data
        """
        suppliers_data = []
        
        compliance_certs = [
            'ISO 9001', 'ISO 14001', 'Six Sigma', 
            'REACH', 'Fair Trade', 'LEED', 'OHSAS 18001'
        ]
        
        for _ in range(self.num_suppliers):
            supplier_id = f'SUP_{str(uuid.uuid4())[:8]}'
            
            suppliers_data.append({
                'supplier_id': supplier_id,
                'name': f'Supplier {supplier_id}',
                'contact_person': f'Contact {random.randint(1, 100)}',
                'location': random.choice(self.regions),
                
                # Ratings and Scores
                'price_rating': round(random.uniform(1, 10), 2),
                'quality_rating': round(random.uniform(1, 10), 2),
                'delivery_time': round(random.uniform(5, 45), 2),
                'reliability_score': round(random.uniform(0, 1), 2),
                'financial_stability_score': round(random.uniform(1, 10), 2),
                
                # Order Constraints
                'min_order_quantity': random.randint(100, 1000),
                'max_order_quantity': random.randint(1000, 10000),
                
                # Performance Metrics
                'lead_time_variability': round(random.uniform(1, 10), 2),
                'production_capacity': random.randint(10000, 100000),
                'transportation_cost': round(random.uniform(50, 500), 2),
                
                # Compliance and Certifications
                'compliance_certifications': random.sample(
                    compliance_certs, 
                    k=random.randint(1, 3)
                ),
                
                # Risk Indicators
                'geopolitical_risk_factor': round(random.uniform(0, 1), 2),
                'environmental_sustainability_score': round(random.uniform(0, 1), 2)
            })
        
        return pd.DataFrame(suppliers_data)
    
    def generate_inventory_data(self):
        """
        Generate comprehensive inventory dataset
        
        :return: Pandas DataFrame with inventory data
        """
        inventory_data = []
        
        for _ in range(self.num_products):
            product_id = f'PROD_{str(uuid.uuid4())[:8]}'
            category = random.choice(self.product_categories)
            base_annual_demand = random.randint(5000, 50000)
            
            inventory_data.append({
                'product_id': product_id,
                'category': category,
                
                # Current Stock
                'current_stock': random.randint(500, 5000),
                'safety_stock_level': random.randint(100, 1000),
                'reorder_point': random.randint(200, 2000),
                'economic_order_quantity': random.randint(300, 3000),
                
                # Demand Characteristics
                'annual_demand': base_annual_demand,
                'daily_average_demand': base_annual_demand / 365,
                'demand_variability': round(random.uniform(0.1, 0.5), 2),
                
                # Cost Factors
                'order_cost': round(random.uniform(10, 200), 2),
                'holding_cost': round(random.uniform(0.1, 5), 2),
                'storage_cost': round(random.uniform(1, 50), 2),
                'stockout_cost': round(random.uniform(50, 500), 2),
                
                # Time-related Metrics
                'lead_time': round(random.uniform(5, 45), 2),
                'lead_time_std': round(random.uniform(1, 10), 2),
                'demand_std': base_annual_demand * round(random.uniform(0.05, 0.2), 2),
                
                # Product Lifecycle and Risk
                'perishability': round(random.uniform(0, 1), 2),
                'shelf_life_days': random.randint(30, 365),
                'obsolescence_risk': round(random.uniform(0, 1), 2)
            })
        
        return pd.DataFrame(inventory_data)
    
    def generate_external_factors_data(self):
        """
        Generate comprehensive external factors dataset
        
        :return: Pandas DataFrame with external factors data
        """
        external_factors_data = []
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='M')
        
        economic_indicators = [
            'GDP Growth Rate', 'Inflation Rate', 'Unemployment Rate', 
            'Consumer Price Index', 'Industrial Production Index'
        ]
        
        for date in dates:
            external_factors_data.append({
                'date': date,
                
                # Economic Indicators
                'economic_indicators': {
                    indicator: round(random.uniform(0, 10), 2) 
                    for indicator in economic_indicators
                },
                
                # Market Sentiment
                'market_sentiment_index': round(random.uniform(50, 150), 2),
                
                # Commodity and Resource Prices
                'commodity_price_index': round(random.uniform(50, 200), 2),
                
                # Risk Factors
                'geopolitical_risk_index': round(random.uniform(0, 10), 2),
                'transportation_cost_index': round(random.uniform(80, 120), 2),
                
                # Climate and Environmental Factors
                'weather_impact_factor': round(random.uniform(-1, 1), 2),
                'natural_disaster_risk': round(random.uniform(0, 1), 2)
            })
        
        return pd.DataFrame(external_factors_data)
    
    def generate_product_master_data(self):
        """
        Generate comprehensive product master dataset
        
        :return: Pandas DataFrame with product master data
        """
        product_master_data = []
        
        for _ in range(self.num_products):
            product_id = f'PROD_{str(uuid.uuid4())[:8]}'
            category = random.choice(self.product_categories)
            
            unit_cost = round(random.uniform(10, 500), 2)
            selling_price = round(unit_cost * random.uniform(1.2, 3), 2)
            
            product_master_data.append({
                'product_id': product_id,
                'product_name': f'{category} Product {product_id}',
                
                # Classification
                'category': category,
                'subcategory': f'Subcategory {random.randint(1, 10)}',
                
                # Cost and Pricing
                'unit_cost': unit_cost,
                'selling_price': selling_price,
                'profit_margin': round((selling_price - unit_cost) / selling_price * 100, 2),
                
                # Production Characteristics
                'manufacturing_lead_time': round(random.uniform(1, 30), 2),
                'production_capacity': random.randint(1000, 10000),
                
                # Physical Attributes
                'weight': round(random.uniform(0.1, 50), 2),
                'dimensions': {
                    'length': round(random.uniform(10, 100), 2),
                    'width': round(random.uniform(10, 100), 2),
                    'height': round(random.uniform(10, 100), 2)
                },
                
                # Lifecycle Management
                'shelf_life_days': random.randint(30, 365),
                'replacement_lead_time': round(random.uniform(7, 90), 2),
                
                # Supply Chain Specific
                'primary_supplier_id': f'SUP_{str(uuid.uuid4())[:8]}',
                'alternative_suppliers': [
                    f'SUP_{str(uuid.uuid4())[:8]}' for _ in range(random.randint(1, 3))
                ]
            })
        
        return pd.DataFrame(product_master_data)
    
    def generate_all_datasets(self, output_dir='./supply_chain_data'):
        """
        Generate and save all datasets
        
        :param output_dir: Directory to save generated datasets
        :return: Dictionary of generated datasets
        """
        import os
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate datasets
        datasets = {
            'sales_data': self.generate_sales_data(),
            'suppliers_data': self.generate_suppliers_data(),
            'inventory_data': self.generate_inventory_data(),
            'external_factors_data': self.generate_external_factors_data(),
            'product_master_data': self.generate_product_master_data()
        }
        
        # Save datasets
        for name, df in datasets.items():
            csv_path = os.path.join(output_dir, f'{name}.csv')
            df.to_csv(csv_path, index=False)
            print(f"Generated {name}: {csv_path}")
        
        return datasets

# Demonstrate usage
def main():
    # Create data generator
    data_generator = AdvancedSupplyChainDataGenerator(
        num_products=50,  # Generate 50 unique products
        num_suppliers=20  # Generate 20 unique suppliers
    )
    
    # Generate and save all datasets
    datasets = data_generator.generate_all_datasets()
    
    # Optional: Preview generated data
    for name, df in datasets.items():
        print(f"\n{name.upper()} Preview:")
        print(df.head())
        print(f"Total rows: {len(df)}")

if __name__ == '__main__':
    main()
