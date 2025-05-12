import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import pymongo
import json
import os
import logging
from datetime import datetime, timedelta
import pulp
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SupplyChainOptimizer:
    def __init__(self, mongodb_uri='mongodb://localhost:27017/', db_name='supply_chain_db'):
        """
        Advanced Supply Chain Optimization System
        
        :param mongodb_uri: MongoDB connection URI
        :param db_name: Name of the MongoDB database
        """
        try:
            # MongoDB Connection
            self.client = pymongo.MongoClient(mongodb_uri)
            self.db = self.client[db_name]
            
            # Collections
            self.sales_collection = self.db['sales_data']
            self.suppliers_collection = self.db['suppliers']
            self.inventory_collection = self.db['inventory']
            self.external_factors_collection = self.db['external_factors']
            self.product_master_collection = self.db['product_master']
            
            # Logging
            logger.info("MongoDB connection established successfully")
        except Exception as e:
            logger.error(f"Failed to establish MongoDB connection: {e}")
            raise
        
        # Machine Learning Models Cache
        self.ml_models = {}
    
    def preprocess_data(self, collection_name, data_type='sales'):
        """
        Advanced data preprocessing with multiple techniques
        
        :param collection_name: Name of the MongoDB collection
        :param data_type: Type of data for specific preprocessing
        :return: Preprocessed DataFrame
        """
        try:
            # Fetch data from MongoDB
            cursor = self.db[collection_name].find()
            df = pd.DataFrame(list(cursor))
            
            if df.empty:
                logger.warning(f"No data found in {collection_name}")
                return None
            
            # Data cleaning strategies
            if data_type == 'sales':
                # Handle sales-specific preprocessing
                df['date'] = pd.to_datetime(df['date'])
                
                # Feature engineering
                df['month'] = df['date'].dt.month
                df['quarter'] = df['date'].dt.quarter
                df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)
                
                # Handle missing values
                numeric_cols = ['sales_quantity', 'sales_value']
                categorical_cols = ['category', 'region', 'channel']
                
                # Imputation pipeline
                numeric_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='median'))
                ])
                
                categorical_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ])
                
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', numeric_transformer, numeric_cols),
                        ('cat', categorical_transformer, categorical_cols)
                    ])
                
                # Normalize numerical features
                scaler = MinMaxScaler()
                df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            
            return df
        except Exception as e:
            logger.error(f"Data preprocessing error: {e}")
            return None
    
    def advanced_demand_forecasting(self, product_id=None, forecast_horizon=90):
        """
        Ensemble demand forecasting with multiple methods
        
        :param product_id: Specific product to forecast
        :param forecast_horizon: Days to forecast
        :return: Forecast results dictionary
        """
        try:
            # Fetch sales data
            query = {'product_id': product_id} if product_id else {}
            sales_data = list(self.sales_collection.find(query))
            df = pd.DataFrame(sales_data)
            
            if df.empty:
                logger.warning("No sales data available for forecasting")
                return None
            
            # Prepare data for forecasting
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Multiple forecasting methods
            forecasts = {}
            
            # 1. Prophet Forecast
            prophet_df = df[['date', 'sales_quantity']].rename(columns={'date': 'ds', 'sales_quantity': 'y'})
            prophet_model = Prophet(
                seasonality_mode='multiplicative',
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False
            )
            prophet_model.fit(prophet_df)
            
            future = prophet_model.make_future_dataframe(periods=forecast_horizon)
            prophet_forecast = prophet_model.predict(future)
            
            forecasts['prophet'] = {
                'forecast': prophet_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
                'model_metrics': {
                    'mape': np.mean(np.abs((prophet_forecast['y'] - prophet_forecast['yhat']) / prophet_forecast['y'])) * 100
                }
            }
            
            # 2. ARIMA Forecast
            arima_model = ARIMA(df['sales_quantity'], order=(5,1,0))
            arima_results = arima_model.fit()
            arima_forecast = arima_results.forecast(steps=forecast_horizon)
            
            forecasts['arima'] = {
                'forecast': arima_forecast,
                'model_metrics': {
                    'aic': arima_results.aic,
                    'bic': arima_results.bic
                }
            }
            
            # 3. Machine Learning Forecast
            features = ['month', 'quarter', 'category_encoded']
            target = 'sales_quantity'
            
            # Prepare ML dataset
            df['month'] = df['date'].dt.month
            df['quarter'] = df['date'].dt.quarter
            df['category_encoded'] = pd.Categorical(df['category']).codes
            
            X = df[features]
            y = df[target]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            
            ml_model = RandomForestRegressor(n_estimators=100)
            ml_model.fit(X_train, y_train)
            
            ml_forecast = ml_model.predict(X_test)
            
            forecasts['ml_model'] = {
                'forecast': ml_forecast,
                'model_metrics': {
                    'r2_score': ml_model.score(X_test, y_test),
                    'feature_importance': dict(zip(features, ml_model.feature_importances_))
                }
            }
            
            return forecasts
        
        except Exception as e:
            logger.error(f"Demand forecasting error: {e}")
            return None
    
    def advanced_supplier_selection(self, criteria_weights=None):
        """
        Comprehensive supplier selection with multi-criteria analysis
        
        :param criteria_weights: Custom weights for supplier evaluation
        :return: Ranked and analyzed suppliers
        """
        try:
            # Fetch suppliers data
            suppliers = list(self.suppliers_collection.find())
            df = pd.DataFrame(suppliers)
            
            # Default criteria weights
            if criteria_weights is None:
                criteria_weights = {
                    'price_rating': 0.25,
                    'quality_rating': 0.25,
                    'delivery_time': 0.2,
                    'reliability_score': 0.2,
                    'financial_stability_score': 0.1
                }
            
            # Normalize and score suppliers
            scaler = StandardScaler()
            criteria = list(criteria_weights.keys())
            
            # Normalize selected criteria
            normalized_df = pd.DataFrame(
                scaler.fit_transform(df[criteria]),
                columns=criteria
            )
            
            # Weighted scoring
            normalized_df['total_score'] = sum(
                normalized_df[criteria] * weight 
                for criteria, weight in criteria_weights.items()
            ).sum(axis=1)
            
            # Risk classification
            risk_classifier = RandomForestClassifier(n_estimators=100)
            risk_features = ['price_rating', 'delivery_time', 'financial_stability_score']
            
            X = df[risk_features]
            y = (df['reliability_score'] < 0.5).astype(int)  # Binary risk classification
            
            risk_classifier.fit(X, y)
            
            # Predict supplier risks
            normalized_df['risk_probability'] = risk_classifier.predict_proba(X)[:, 1]
            
            # Final supplier ranking
            final_ranking = normalized_df.sort_values('total_score', ascending=False)
            
            return {
                'supplier_ranking': final_ranking,
                'risk_analysis': {
                    'high_risk_suppliers': final_ranking[final_ranking['risk_probability'] > 0.5],
                    'low_risk_suppliers': final_ranking[final_ranking['risk_probability'] <= 0.5]
                }
            }
        
        except Exception as e:
            logger.error(f"Supplier selection error: {e}")
            return None
    
    def advanced_inventory_optimization(self, product_id=None):
        """
        Advanced inventory optimization with multiple strategies
        
        :param product_id: Specific product to optimize
        :return: Inventory optimization recommendations
        """
        try:
            # Fetch inventory and sales data
            query = {'product_id': product_id} if product_id else {}
            inventory_data = list(self.inventory_collection.find(query))
            df = pd.DataFrame(inventory_data)
            
            if df.empty:
                logger.warning("No inventory data available")
                return None
            
            # Linear Programming for Inventory Optimization
            prob = pulp.LpProblem("Inventory_Optimization", pulp.LpMinimize)
            
            # Decision Variables
            order_quantities = {
                row['product_id']: pulp.LpVariable(f"order_{row['product_id']}", lowBound=0)
                for _, row in df.iterrows()
            }
            
            # Objective Function: Minimize Total Cost
            prob += pulp.lpSum([
                (row['order_cost'] + row['holding_cost'] * order_quantities[row['product_id']])
                for _, row in df.iterrows()
            ])
            
            # Constraints
            for _, row in df.iterrows():
                # Demand Satisfaction
                prob += order_quantities[row['product_id']] >= row['annual_demand']
                
                # Capacity Constraints
                prob += order_quantities[row['product_id']] <= row['max_order_quantity']
            
            # Solve the problem
            prob.solve()
            
            # Optimization Results
            optimization_results = {
                'status': pulp.LpStatus[prob.status],
                'optimal_orders': {
                    product_id: order_quantities[product_id].varValue
                    for product_id in order_quantities
                },
                'total_cost': pulp.value(prob.objective)
            }
            
            return optimization_results
        
        except Exception as e:
            logger.error(f"Inventory optimization error: {e}")
            return None
    
    def supply_chain_risk_assessment(self):
        """
        Comprehensive supply chain risk assessment
        
        :return: Detailed risk analysis
        """
        try:
            # Fetch relevant data
            suppliers = list(self.suppliers_collection.find())
            inventory = list(self.inventory_collection.find())
            external_factors = list(self.external_factors_collection.find())
            
            # Risk Assessment Components
            risk_assessment = {
                'supplier_risks': [],
                'inventory_risks': [],
                'external_risks': []
            }
            
            # Supplier Risks
            for supplier in suppliers:
                risk_score = 0
                if supplier['reliability_score'] < 0.5:
                    risk_score += 0.3
                if supplier['delivery_time'] > 30:
                    risk_score += 0.2
                
                risk_assessment['supplier_risks'].append({
                    'supplier_id': supplier['supplier_id'],
                    'name': supplier['name'],
                    'risk_score': risk_score
                })
            
            # Inventory Risks
            for item in inventory:
                risk_score = 0
                if item['current_stock'] < item['safety_stock_level']:
                    risk_score += 0.4
                if item['perishability'] > 0.7:
                    risk_score += 0.3
                
                risk_assessment['inventory_risks'].append({
                    'product_id': item['product_id'],
                    'risk_score': risk_score
                })
            
            # External Risks
            for factor in external_factors:
                risk_score = 0
                if factor['geopolitical_risk'] > 7:
                    risk_score += 0.3
                if factor['transportation_cost_index'] > 110:
                    risk_score += 0.2
                
                risk_assessment['external_risks'].append({
                    'date': factor['date'],
                    'risk_score': risk_score
                })
            
            # Overall Risk Classification
            risk_assessment['overall_risk_level'] = np.mean([
                np.mean([r['risk_score'] for r in risk_assessment['supplier_risks']]),
                np.mean([r['risk_score'] for r in risk_assessment['inventory_risks']]),
                np.mean([r['risk_score'] for r in risk_assessment['external_risks']])
            ])
            
            return risk_assessment
        
        except Exception as e:
            logger.error(f"Risk assessment error: {e}")
            return None
    
    def generate_comprehensive_report(self, report_format='json', output_dir='./supply_chain_reports'):
        """
        Generate a comprehensive supply chain report with multiple output formats
        
        :param report_format: Output format (json, pdf, markdown, html)
        :param output_dir: Directory to save generated reports
        :return: Report file path or report content
        """
        try:
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Collect insights from different modules
            report_data = {
                'report_id': str(uuid.uuid4()),
                'generated_at': datetime.now().isoformat(),
                'demand_forecast': self.advanced_demand_forecasting(),
                'supplier_analysis': self.advanced_supplier_selection(),
                'inventory_optimization': self.advanced_inventory_optimization(),
                'risk_assessment': self.supply_chain_risk_assessment()
            }
            
            # Generate report based on format
            report_path = os.path.join(output_dir, f'supply_chain_report_{report_data["report_id"]}')
            
            if report_format == 'json':
                full_path = f'{report_path}.json'
                with open(full_path, 'w') as f:
                    json.dump(report_data, f, indent=2)
                return full_path
            
            elif report_format == 'markdown':
                full_path = f'{report_path}.md'
                with open(full_path, 'w') as f:
                    # Markdown Report Generation
                    f.write(f"# Supply Chain Comprehensive Report\n")
                    f.write(f"**Report ID:** {report_data['report_id']}\n")
                    f.write(f"**Generated At:** {report_data['generated_at']}\n\n")
                    
                    # Demand Forecast Section
                    f.write("## Demand Forecast Analysis\n")
                    if report_data['demand_forecast']:
                        for method, forecast in report_data['demand_forecast'].items():
                            f.write(f"### {method.upper()} Forecast\n")
                            f.write("**Model Metrics:**\n")
                            for metric, value in forecast.get('model_metrics', {}).items():
                                f.write(f"- {metric.upper()}: {value}\n")
                    
                    # Supplier Analysis Section
                    f.write("\n## Supplier Analysis\n")
                    if report_data['supplier_analysis']:
                        f.write("### Supplier Ranking\n")
                        f.write("| Supplier | Total Score | Risk Probability |\n")
                        f.write("|----------|-------------|------------------|\n")
                        for _, row in report_data['supplier_analysis']['supplier_ranking'].iterrows():
                            f.write(f"| {row.name} | {row['total_score']:.2f} | {row.get('risk_probability', 0):.2f} |\n")
                    
                    # Inventory Optimization Section
                    f.write("\n## Inventory Optimization\n")
                    if report_data['inventory_optimization']:
                        f.write(f"**Optimization Status:** {report_data['inventory_optimization']['status']}\n")
                        f.write(f"**Total Cost:** {report_data['inventory_optimization']['total_cost']:.2f}\n")
                    
                    # Risk Assessment Section
                    f.write("\n## Supply Chain Risk Assessment\n")
                    if report_data['risk_assessment']:
                        f.write(f"**Overall Risk Level:** {report_data['risk_assessment']['overall_risk_level']:.2f}\n")
                        
                        f.write("\n### Supplier Risks\n")
                        for risk in report_data['risk_assessment']['supplier_risks']:
                            f.write(f"- **{risk['name']}**: Risk Score {risk['risk_score']:.2f}\n")
                        
                        f.write("\n### Inventory Risks\n")
                        for risk in report_data['risk_assessment']['inventory_risks']:
                            f.write(f"- **Product {risk['product_id']}**: Risk Score {risk['risk_score']:.2f}\n")
                
                return full_path
            
            elif report_format == 'html':
                full_path = f'{report_path}.html'
                with open(full_path, 'w') as f:
                    # HTML Report Generation
                    f.write("<!DOCTYPE html>\n<html lang='en'>\n<head>")
                    f.write("<meta charset='UTF-8'>")
                    f.write("<title>Supply Chain Comprehensive Report</title>")
                    f.write("<style>")
                    f.write("body { font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }")
                    f.write("h1, h2 { color: #333; }")
                    f.write("table { width: 100%; border-collapse: collapse; margin-bottom: 20px; }")
                    f.write("table, th, td { border: 1px solid #ddd; padding: 8px; }")
                    f.write("</style>")
                    f.write("</head>\n<body>")
                    
                    f.write(f"<h1>Supply Chain Comprehensive Report</h1>")
                    f.write(f"<p><strong>Report ID:</strong> {report_data['report_id']}</p>")
                    f.write(f"<p><strong>Generated At:</strong> {report_data['generated_at']}</p>")
                    
                    # Similar sections as Markdown, but with HTML formatting
                    f.write("</body>\n</html>")
                
                return full_path
            
            elif report_format == 'pdf':
                try:
                    from reportlab.lib.pagesizes import letter
                    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
                    from reportlab.lib.styles import getSampleStyleSheet
                    
                    full_path = f'{report_path}.pdf'
                    doc = SimpleDocTemplate(full_path, pagesize=letter)
                    styles = getSampleStyleSheet()
                    story = []
                    
                    # Add report content similar to markdown/html, but using ReportLab
                    story.append(Paragraph("Supply Chain Comprehensive Report", styles['Title']))
                    story.append(Paragraph(f"Report ID: {report_data['report_id']}", styles['Normal']))
                    story.append(Paragraph(f"Generated At: {report_data['generated_at']}", styles['Normal']))
                    story.append(Spacer(1, 12))
                    
                    # Add more PDF-specific report generation logic
                    doc.build(story)
                    
                    return full_path
                except ImportError:
                    logger.warning("ReportLab not installed. Falling back to JSON.")
                    full_path = f'{report_path}.json'
                    with open(full_path, 'w') as f:
                        json.dump(report_data, f, indent=2)
                    return full_path
            
            else:
                raise ValueError(f"Unsupported report format: {report_format}")
        
        except Exception as e:
            logger.error(f"Report generation error: {e}")
            return None
