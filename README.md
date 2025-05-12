# Supply Chain Expert System: Mathematical and Computational Optimization

## 1. Demand Forecasting

### Mathematical Model

#### Seasonal Decomposition Mathematics

**Yearly Seasonality Model**:
$S_{yearly} = A \cdot \sin\left(\frac{2\pi \cdot month}{12}\right)$

Where:
- $S_{yearly}$ is the yearly seasonal component
- $A$ is the amplitude (scaling factor)
- $month$ represents the current month (1-12)

**Weekly Seasonality Model**:
$S_{weekly} = B \cdot \sin\left(\frac{2\pi \cdot day}{7}\right)$

Where:
- $S_{weekly}$ is the weekly seasonal component
- $B$ is the amplitude (scaling factor)
- $day$ represents the day of the week (0-6)

**Composite Seasonal Forecast**:
$Forecast = Base_{demand} + S_{yearly} + S_{weekly} + \epsilon$

**Error Metrics**:
1. Mean Absolute Percentage Error (MAPE):
   $MAPE = \frac{100\%}{n} \sum_{t=1}^{n} \left|\frac{Actual_t - Forecast_t}{Actual_t}\right|$

### Implementation Code

```python
def advanced_demand_forecasting(self, product_id=None, forecast_horizon=90):
    # Prophet Forecast with Seasonal Decomposition
    prophet_model = Prophet(
        seasonality_mode='multiplicative',
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    
    # Prepare data
    prophet_df = df[['date', 'sales_quantity']].rename(
        columns={'date': 'ds', 'sales_quantity': 'y'}
    )
    
    # Fit and forecast
    prophet_model.fit(prophet_df)
    future = prophet_model.make_future_dataframe(periods=forecast_horizon)
    prophet_forecast = prophet_model.predict(future)
    
    # Calculate MAPE
    mape = np.mean(np.abs(
        (prophet_forecast['y'] - prophet_forecast['yhat']) / 
        prophet_forecast['y']
    )) * 100
    
    return {
        'forecast': prophet_forecast,
        'mape': mape
    }
```

## 2. Supplier Selection

### Mathematical Model

**Normalization Function**:
$X_{normalized,i} = \frac{X_i - \min(X)}{\max(X) - \min(X)}$

**Weighted Scoring Function**:
$Total_{Score} = \sum_{i=1}^{n} (X_{normalized,i} \cdot W_i)$

**Risk Probability Calculation**:
$Risk_{Probability} = \sigma\left(\sum_{j=1}^{m} \beta_j X_j\right)$

Where:
- $\sigma$ is the sigmoid function
- $\beta_j$ are model coefficients
- $X_j$ are risk features
- $m$ is the number of risk features

### Implementation Code

```python
def advanced_supplier_selection(self, criteria_weights=None):
    # Default criteria weights
    if criteria_weights is None:
        criteria_weights = {
            'price_rating': 0.25,
            'quality_rating': 0.25,
            'delivery_time': 0.2,
            'reliability_score': 0.2,
            'financial_stability_score': 0.1
        }
    
    # Normalize supplier metrics
    scaler = StandardScaler()
    criteria = list(criteria_weights.keys())
    normalized_df = pd.DataFrame(
        scaler.fit_transform(df[criteria]),
        columns=criteria
    )
    
    # Calculate weighted total score
    normalized_df['total_score'] = sum(
        normalized_df[criteria] * weight 
        for criteria, weight in criteria_weights.items()
    ).sum(axis=1)
    
    # Risk classification
    risk_classifier = RandomForestClassifier(n_estimators=100)
    risk_features = ['price_rating', 'delivery_time', 'financial_stability_score']
    
    X = df[risk_features]
    y = (df['reliability_score'] < 0.5).astype(int)
    
    risk_classifier.fit(X, y)
    normalized_df['risk_probability'] = risk_classifier.predict_proba(X)[:, 1]
    
    return {
        'supplier_ranking': normalized_df.sort_values('total_score', ascending=False),
        'risk_analysis': {
            'high_risk_suppliers': normalized_df[normalized_df['risk_probability'] > 0.5],
            'low_risk_suppliers': normalized_df[normalized_df['risk_probability'] <= 0.5]
        }
    }
```

## 3. Inventory Optimization

### Mathematical Model

**Total Cost Minimization**:
$\min Z = \sum_{i=1}^{n} (O_i \cdot C_{order} + H_i \cdot C_{holding})$

Subject to constraints:
1. Demand Satisfaction: $O_i \geq D_i$
2. Capacity Constraint: $O_i \leq Q_{max,i}$

Where:
- $Z$ is the total cost
- $O_i$ is the order quantity for product $i$
- $C_{order}$ is the order cost
- $C_{holding}$ is the holding cost
- $D_i$ is the annual demand
- $Q_{max,i}$ is the maximum order quantity
- $n$ is the number of products

### Implementation Code

```python
def advanced_inventory_optimization(self, product_id=None):
    # Fetch inventory data
    query = {'product_id': product_id} if product_id else {}
    inventory_data = list(self.inventory_collection.find(query))
    df = pd.DataFrame(inventory_data)
    
    # Linear Programming Optimization
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
    
    return {
        'status': pulp.LpStatus[prob.status],
        'optimal_orders': {
            product_id: order_quantities[product_id].varValue
            for product_id in order_quantities
        },
        'total_cost': pulp.value(prob.objective)
    }
```

## 4. Risk Assessment

### Mathematical Model

**Individual Risk Component**:
$R_i = \sum_{j=1}^{k} w_j \cdot f_j(x_i)$

**Composite Risk Level Calculation**:
$Risk_{Overall} = \frac{1}{3}\left(\frac{\sum R_{supplier}}{n_{supplier}} + \frac{\sum R_{inventory}}{n_{inventory}} + \frac{\sum R_{external}}{n_{external}}\right)$

### Implementation Code

```python
def supply_chain_risk_assessment(self):
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
    
    # Supplier Risks Calculation
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
    
    # Inventory Risks Calculation
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
    
    # External Risks Calculation
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
    
    # Overall Risk Calculation
    risk_assessment['overall_risk_level'] = np.mean([
        np.mean([r['risk_score'] for r in risk_assessment['supplier_risks']]),
        np.mean([r['risk_score'] for r in risk_assessment['inventory_risks']]),
        np.mean([r['risk_score'] for r in risk_assessment['external_risks']])
    ])
    
    return risk_assessment
```

## 5. Data Preprocessing

### Mathematical Model

**Min-Max Scaling**:
$X_{scaled} = \frac{X - X_{min}}{X_{max} - X_{min}}$

**Z-Score Normalization**:
$Z = \frac{X - \mu}{\sigma}$

### Implementation Code

```python
def preprocess_data(self, collection_name, data_type='sales'):
    # Fetch data from MongoDB
    cursor = self.db[collection_name].find()
    df = pd.DataFrame(list(cursor))
    
    # Data cleaning and feature engineering
    if data_type == 'sales':
        df['date'] = pd.to_datetime(df['date'])
        
        # Temporal feature extraction
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)
        
        # Preprocessing pipeline
        numeric_cols = ['sales_quantity', 'sales_value']
        categorical_cols = ['category', 'region', 'channel']
        
        # Imputation and transformation
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', MinMaxScaler())
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
        
        # Apply transformations
        preprocessed_data = preprocessor.fit_transform(df)
        
        return preprocessed_data
```

## Computational Complexity Analysis

### Complexity Metrics

- Demand Forecasting: $O(n \log n)$
- Supplier Selection: $O(m \cdot n)$
- Inventory Optimization: $O(n^2)$
- Risk Assessment: $O(k \cdot m)$

Where:
- $n$ is the number of products/suppliers
- $m$ is the number of features
- $k$ is the number of risk factors

## Key Insights

The expert system combines:
1. Advanced mathematical modeling
2. Statistical inference techniques
3. Machine learning algorithms
4. Optimization strategies

Each component integrates:
- Probabilistic reasoning
- Quantitative analysis
- Adaptive learning mechanisms

### Practical Applications

The integrated approach enables:
- Precise demand forecasting
- Optimal supplier selection
- Efficient inventory management
- Comprehensive risk mitigation

## Conclusion

The mathematical and computational framework provides a robust, data-driven methodology for supply chain optimization, leveraging sophisticated techniques to generate actionable, strategic insights.
