import streamlit as st
import pandas as pd
import pickle

model_path = r"C:\SupplyChain_PredictionModel\research\best_model.pkl"
preprocessor_path = r"C:\SupplyChain_PredictionModel\research\preprocessor.pkl"

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(preprocessor_path, "rb") as f:
    preprocessor = pickle.load(f)

st.title("Supply Chain Late Delivery Risk Prediction Dashboard")

# Define min, max, step for sliders
numeric_cols = {
    'Days_for_shipping_(real)': (0, 6, 3),                      
    'Days_for_shipment_(scheduled)': (0, 4, 2),                 
    'Benefit_per_order': (-4300.0, 920.0, 22.0),                 
    'Sales_per_customer': (7.0, 1940.0, 183.0),                  
    'Order_Item_Discount': (0.0, 500.0, 20.0),                   
    'Order_Item_Product_Price': (10.0, 2000.0, 141.0),           
    'Order_Item_Profit_Ratio': (-3.0, 1.0, 0.12),                
    'Order_Item_Quantity': (1, 5, 2),                            
    'Sales': (10.0, 2000.0, 203.0),                              
    'Order_Item_Total': (7.0, 1940.0, 183.0),                    
    'Order_Profit_Per_Order': (-4300.0, 920.0, 22.0),            
    'Product_Price': (10.0, 2000.0, 141.0)                       
}

categorical_cols = {
    'Type': ["TRANSFER","CASH","DEBIT","PAYMENT"], 
    'Category_Name': ["Sporting Goods", "Accessories", "As Seen on TV!","Baby","Baseball & Softball",
                      "Basketball","Books","Boxing & MMA","Camping & Hiking","Cameras","Children's Clothing",
                      "CDs","Cleats","Computers","Crafts","DVDs","Electronics","Fishing","Fitness Accessories",
                      "Girls' Apparel","Garden","Golf Apparel","Golf Bags & Carts","Golf Balls","Golf Gloves",
                      "Golf Shoes","Health and Beauty","Hockey","Hunting & Shooting","Indoor/Outdoor Games",
                      "Kids' Golf Clubs","Lacrosse","Men's Clothing","Men's Footwear","Men's Golf Clubs","Music",
                      "Pet Supplies","Shop By Sport","Soccer","Sporting Goods","Strength Training","Tennis & Racquet",
                      "Toys","Trade-In","Video Games","Water Sports","Women's Apparel","Women's Clothing","Women's Golf Clubs"],
    'Department_Name': ["Outdoors"], 
    'Market': ["Europe","LATAM"], 
    'Order_Region': ["Northern Europe","Southern Europe","Western Europe"], 
    'Shipping_Mode': ["Standard Class","Second Class","First Class"]
}

# Collect numeric inputs
numeric_input = {}
valid = True

for col, (min_val, max_val, default) in numeric_cols.items():
    user_value = st.text_input(col, value=str(default))
    
    try:
        num_value = float(user_value)
        
        if not (min_val <= num_value <= max_val):
            st.warning(f"{col} must be between {min_val} and {max_val}")
            valid = False
        else:
            numeric_input[col] = num_value
            
    except ValueError:
        st.warning(f"{col} must be a valid number")
        valid = False

# Only allow prediction if all inputs valid
if valid:
    st.success("All inputs are valid")

# Collect categorical inputs
categorical_input = {}
for col, options in categorical_cols.items():
    categorical_input[col] = st.selectbox(col, options)

input_data = {**numeric_input, **categorical_input}
input_df = pd.DataFrame([input_data])

# Predict
if st.button("Predict"):
    
    # Apply preprocessing FIRST
    input_processed = preprocessor.transform(input_df)
    
    # Then predict
    pred = model.predict(input_processed)

    st.subheader(f"Late Delivery Risk: {'Yes' if pred[0] == 1 else 'No'}")