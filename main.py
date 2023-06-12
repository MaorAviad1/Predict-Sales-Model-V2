import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


class SalesPredictor:
    def __init__(self, sales_data_file, inventory_data_file, product_id):
        # Load sales data
        self.sales_df = pd.read_csv(sales_data_file)
        self.sales_df = self.sales_df[self.sales_df['product_id'] == product_id]

        # Load inventory data
        self.inventory_df = pd.read_csv(inventory_data_file)
        self.inventory_df = self.inventory_df[self.inventory_df['product_id'] == product_id]

        # Merge sales and inventory data on date
        self.df = pd.merge(self.sales_df, self.inventory_df, on=['product_id', 'date'], how='inner')

        # Data preprocessing
        self.df['location'] = self.df['location'].astype('category').cat.codes
        self.df['competitors'] = self.df['competitors'].astype('category').cat.codes
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df['month'] = self.df['date'].dt.month

        # Define features and target
        self.X = self.df[['location', 'competitors', 'discounts', 'holidays', 'shippings', 'end_season_sale', 'month',
                          'stock_level']]
        self.y = self.df['quantity_sold']

        # Initialize model
        self.model = LinearRegression()

    def train(self, test_size=0.2, random_state=0):
        # Split the dataset into a training set and a test set
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size,
                                                                                random_state=random_state)

        # Train the model
        self.model.fit(self.X_train, self.y_train)
        print("Model trained.")

    def predict(self, last_quarter_data):
        # Use the model to predict sales for the last quarter
        prediction = self.model.predict(last_quarter_data)
        print(f"Predicted sales for the last quarter of the year: {prediction.sum()}")


# Usage:
predictor = SalesPredictor('sales_data.csv', 'inventory_data.csv', 'product1')
predictor.train()

last_quarter = pd.DataFrame({
    'location': [1, 1, 1],  # Replace with location code
    'competitors': [1, 1, 1],  # Replace with competitors code
    'discounts': [1, 1, 0],  # Example values
    'holidays': [1, 0, 1],  # Example values
    'shippings': [1, 0, 0],  # Example values
    'end_season_sale': [1, 1, 1],  # Example values
    'month': [10, 11, 12],  # Last quarter months
    'stock_level': [100, 200, 150]  # Example values
})

predictor.predict(last_quarter)
