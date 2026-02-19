import pandas as pd
import numpy as np
from config import Config
# data ingestion
def data_ingestion():
  df = pd.read_csv(Config.filepath)
  df.drop(columns=[
        "Customer_Email",
        "Customer_Password",
        "Customer_Fname",
        "Customer_Lname",
        "Product_Image",
        "Product_Description",
        "Order_Id",
        "Customer_Id",
        'Customer_City',
        'Customer_Country',
        'Customer_Segment',
        'Customer_State',
        'Customer_Street',
        'Customer_Zipcode',
        'Order_City',
        'Order_Country',
        'Order_State',
        'Order_Zipcode',
        'Product_Status',
        "Order_Customer_Id",
        'Category_Id',
        'Latitude',
        'Longitude',
        'Order_Item_Id',
        'Product_Category_Id',
        'shipping_date_(DateOrders)',
        'order_date_(DateOrders)',
        'Product_Card_Id',
        'Order_Item_Cardprod_Id',
        'Department_Id',
        "Delivery_Status",
        "Order_Status",
        "Product_Name",
        'Order_Item_Discount_Rate'
    ],axis=1, inplace=True)
  return df