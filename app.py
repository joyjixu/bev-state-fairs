import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import joblib
import sys
sys.modules['sklearn.externals.joblib'] = joblib

# load dfs
MN_df_scaled = pd.read_csv("MN_df_scaled.csv")
TX_df_scaled = pd.read_csv("TX_df_scaled.csv")
NC_df_scaled = pd.read_csv("NC_df_scaled.csv")
NY_df_scaled = pd.read_csv("NY_df_scaled.csv")

# load scalers
NC_scaler_prec = joblib.load("NC_scaler_prec.save") 
NY_scaler_prec = joblib.load("NY_scaler_prec.save") 
TX_scaler_prec = joblib.load("TX_scaler_prec.save") 
MN_scaler_prec = joblib.load("MN_scaler_prec.save") 

scalers_dict = {'New York':NY_scaler_prec,
'North Carolina':NC_scaler_prec,
'Texas':TX_scaler_prec,
'Minnesota':MN_scaler_prec}

# create functions

def func(t, a, b, c):
    #return a + b * np.log(c*t)
    #return a * 1/(1 + np.exp(b*t+c))
    return a*np.exp(-b*t-c) 

def fit_curve(state_df):
    x = np.array(state_df["scaled_prec"])
    y = np.array(state_df["scaled_att"])
    popt, pcov = curve_fit(func, x, y,maxfev=5000, bounds=(0, np.inf))
    p,a = zip(*sorted(zip(x, func(x, *popt))))
    return popt

def adjust_vendor_revenue(df,precipitation, gross_revenue, popt, scaler_prec):
    
    scaled_prec = scaler_prec.transform(np.array(precipitation).reshape(-1, 1))
    no_rain_att =  func(0, popt[0],popt[1],popt[2]) #returns scaled attendance no rain
    y_pred = func(scaled_prec[0,0], popt[0],popt[1],popt[2]) #returns scaled attendance with rain
    sf = (y_pred)/no_rain_att

    new_revenue = sf*gross_revenue

    return (new_revenue)

def rainfall_profit(df_scaled, popt, rainfall, scaler_prec, daily_revenue, percent_commission=0.20, percent_sales_tax=0.0738,percent_admission=0.02, percent_labour=0.18, percent_food=0.25, percent_income_tax = 0.12, extra_costs=0):
    adjusted_rev = adjust_vendor_revenue(df_scaled, rainfall,daily_revenue, popt, scaler_prec)
    commission = percent_commission*adjusted_rev
    sales_tax = percent_sales_tax*adjusted_rev

    admission = percent_admission*daily_revenue
    labour = percent_labour*daily_revenue
    food_cost = percent_food*daily_revenue

    pre_tax_profit = adjusted_rev - commission - sales_tax - admission - labour - food_cost - extra_costs


    calc_income_tax = lambda a : a*percent_income_tax if a > 0 else 0
    income_tax = calc_income_tax(pre_tax_profit)

    net_profit = pre_tax_profit - income_tax
    return net_profit

def total_profit_fair(df_scaled, popt, scaler_prec, total_revenue, prec_list, percent_commission=0.20, percent_sales_tax=0.0738,percent_admission=0.02, percent_labour=0.18, percent_food=0.25, percent_income_tax = 0.12, extra_costs=0):
    profit_list = []
    daily_revenue = total_revenue/len(prec_list)
    for prec in prec_list:
        profit_list.append(rainfall_profit(df_scaled, popt,prec, scaler_prec, daily_revenue, percent_commission=percent_commission, percent_sales_tax=percent_sales_tax,percent_admission=percent_admission, percent_labour=percent_labour, percent_food=percent_food, percent_income_tax = percent_income_tax, extra_costs=extra_costs))
    
    rainy_days = len([x for x in prec_list if x > 0])

    return prec_list, profit_list, rainy_days


def total_profit_fair_us(total_revenue, scalers_dict, prec_list, percent_commission=0.20, percent_sales_tax=0.0738,percent_admission=0.02, percent_labour=0.18, percent_food=0.25, percent_income_tax = 0.12,extra_costs=0):
    profit_list_MN = []
    profit_list_TX = []
    profit_list_NC = []
    profit_list_NY = []
    daily_revenue = total_revenue/len(prec_list)
    for prec in prec_list:
        profit_list_MN.append(rainfall_profit(MN_df_scaled, popt_MN,prec, scalers_dict['MN'], daily_revenue, percent_commission=percent_commission, percent_sales_tax=percent_sales_tax,percent_admission=percent_admission, percent_labour=percent_labour, percent_food=percent_food, percent_income_tax = percent_income_tax, extra_costs=0))
        profit_list_TX.append(rainfall_profit(TX_df_scaled, popt_TX,prec, scalers_dict['TX'],daily_revenue, percent_commission=percent_commission, percent_sales_tax=percent_sales_tax,percent_admission=percent_admission, percent_labour=percent_labour, percent_food=percent_food, percent_income_tax = percent_income_tax, extra_costs=0))
        profit_list_NC.append(rainfall_profit(NC_df_scaled, popt_NC,prec, scalers_dict['NC'], daily_revenue, percent_commission=percent_commission, percent_sales_tax=percent_sales_tax,percent_admission=percent_admission, percent_labour=percent_labour, percent_food=percent_food, percent_income_tax = percent_income_tax, extra_costs=0))
        profit_list_NY.append(rainfall_profit(NY_df_scaled, popt_NY,prec,scalers_dict['NY'], daily_revenue, percent_commission=percent_commission, percent_sales_tax=percent_sales_tax,percent_admission=percent_admission, percent_labour=percent_labour, percent_food=percent_food, percent_income_tax = percent_income_tax, extra_costs=0))
    avg_profit_list = np.average(np.array([profit_list_MN, profit_list_TX, profit_list_NC, profit_list_NY]), axis=0)
    
    total_profit_MN = sum(profit_list_MN)
    total_profit_TX = sum(profit_list_TX)
    total_profit_NC = sum(profit_list_NC)
    total_profit_NY = sum(profit_list_NY)

    avg_total_profit = (total_profit_MN + total_profit_TX + total_profit_NC + total_profit_NY)/4

    rainy_days = len([x for x in prec_list if x > 0])


    return prec_list, avg_profit_list, rainy_days

def add_insurance(prec_list, profit_list, threshold, payout, premium):
    paid = False
    new_profit_list = profit_list.copy()
    for i in range(len(prec_list)):
        if (prec_list[i]>threshold) and (paid==False):
            new_profit_list[i] = profit_list[i] + payout
            paid = True
        elif (paid==True):
            new_profit_list[i] = profit_list[i]
        elif (paid==False):
            new_profit_list[i] = profit_list[i] - premium

    return new_profit_list


popt_NC = fit_curve(NC_df_scaled)
popt_MN = fit_curve(MN_df_scaled)
popt_NY = fit_curve(NY_df_scaled)
popt_TX = fit_curve(TX_df_scaled)

# plotting functions
def plotly_prec_profit(prec_list, profit_list):

    fig = make_subplots(rows=2, cols=1,
                    shared_xaxes=True)
    

    fig.add_trace(
        go.Bar(x=[x for x in range(1,len(prec_list)+1)], y=prec_list,name="Precipitation"),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(x=[x for x in range(1,len(profit_list)+1)], y=profit_list, name = "Profit"),
        row=2, col=1
    )


    fig.update_layout(title_text="Daily Precipitation and Profit",)

    # Update xaxis properties
    fig.update_xaxes(title_text="Day",row=1, col=1)
    fig.update_xaxes(title_text="Day", row=2, col=1)


    # Update yaxis properties
    fig.update_yaxes(title_text="Precipitation (mm)", row=1, col=1)
    fig.update_yaxes(title_text="Expected Profit (USD)", row=2, col=1)

    return fig

# UI helper functions
def select_state_df(state):
    state_dict = {'Texas':TX_df_scaled,
                'Minnesota':MN_df_scaled,
                'North Carolina':NC_df_scaled,
                'New York':NY_df_scaled}
    return state_dict[state]

def select_state_popt(state):
    state_dict = {'Texas':popt_TX,
                'Minnesota':popt_MN,
                'North Carolina':popt_NC,
                'New York':popt_NY}    
    return state_dict[state]   


# text and display
st.title('US State Fair Rainfall Simulation 🌧')
st.write('This predictive model  adjusts the revenue of US state fair vendors according to unexpected rainfall, as part of the TKE Consultancy project performed by UCL students for BirdsEyeView Technologies.')

st.header('1. Select your variables 🔧')  
st.write("Set up the parameters of the model.")

location = st.selectbox('Choose a location: ', ('Texas', 'Minnesota', 'North Carolina', 'New York', 'US'),help='Choose a specific state or an average for US-wide predictions.')
revenue = st.number_input('Choose the total revenue (whole fair): ', min_value=1, value=50000, help="This is the amount of money (USD) the vendor expects to earn over the whole fair duration, under normal weather circumstances.")

custom_days = st.checkbox('Custom fair duration?', help="Check if you would like a different duration than the dataset average.(12 days)")
if custom_days:
    days = st.slider('Choose the fair duration (days): ', min_value=1, max_value=30, value=12)
else:
    days = 12

custom_expenses_needed = st.checkbox('Adjust custom expenses?', help="Check if you would like to change the expenses of a vendor (e.g. commission) from the average costs based on the literature.")
if custom_expenses_needed:
    percent_commission=st.number_input('Percent commission', min_value=0.0, value=0.20, format="%g")
    percent_sales_tax=st.number_input('Percent sales tax', min_value=0.0, value=0.0738, format="%g")
    percent_admission=st.number_input('Percent admission tickets', min_value=0.0, value=0.02, format="%g")
    percent_labour=st.number_input('Percent labour costs', min_value=0.0, value=0.18, format="%g")
    percent_food=st.number_input('Percent food and raw material costs', min_value=0.0, value=0.25, format="%g")
    percent_income_tax = st.number_input('Percent income tax', min_value=0.0, value=0.12, format="%g")
else:
    percent_commission=0.20
    percent_sales_tax=0.0738
    percent_admission=0.02
    percent_labour=0.18
    percent_food=0.25
    percent_income_tax = 0.12
    
extra_costs_needed = st.checkbox('Extra costs?', help="We assume there are no extra costs by default, check if you would like to account for a one-off expense (e.g. equipment).")
if extra_costs_needed:
    extra_costs = st.number_input('How much is paid in extra costs (USD over the whole fair)?', min_value=0)
else:
    extra_costs = 0

st.header('2. Enter precipitation 💧')
st.write("For each day, enter the mm of rainfall.")
prec_list = []
for day in range(1,days+1):
    prec_day = st.number_input('Select precipitation for day {} (mm):'.format(day), min_value=0)
    prec_list.append(prec_day)


st.header('3. Results 🔍')
st.write('Predictions for how profit would be affected by rainfall, considering the expenses of the vendor as well.')
if location == 'US':
    prec_list, profit_list, rainy_days = total_profit_fair_us(revenue, scalers_dict, prec_list, percent_commission=percent_commission, percent_sales_tax=percent_sales_tax,percent_admission=percent_admission, percent_labour=percent_labour, percent_food=percent_food, percent_income_tax = percent_income_tax, extra_costs=extra_costs)

else:
    prec_list, profit_list, rainy_days = total_profit_fair(select_state_df(location), select_state_popt(location), scalers_dict[location], revenue, prec_list, percent_commission=percent_commission, percent_sales_tax=percent_sales_tax,percent_admission=percent_admission, percent_labour=percent_labour, percent_food=percent_food, percent_income_tax = percent_income_tax, extra_costs=extra_costs)

st.write('Chosen location: {}'.format(location))
output_str = ('Total days: {}'.format(len(prec_list))) + ('  \n Rainy days: {}'.format(rainy_days)) + ('  \n Total Profit: {}'.format(sum(profit_list)))
st.markdown(output_str)

st.subheader("Precipitation and vendor profit over fair duration")
st.plotly_chart(plotly_prec_profit(prec_list, profit_list))

st.header('4. Insurance 📝')
st.write("Below, you can add the option of purchasing BirdsEyeView's parametric weather risk insurance product, which benefits the vendor in the case where extreme weather causes a drastic loss in profits.")

threshold = st.number_input('Select threshold for payout (mm):', min_value=0, help="This is the minimum mm of rain that must fall for the payout to be given.")
payout = st.number_input('Choose payout value (USD):', min_value=0, help="This is how much the vendor will recieve id the threshold is reached.")
premium = st.number_input('Choose daily premium (USD):', min_value=0, help="This is the cost of purchasing BirdsEyeView's insurance, per day until the end of the fair or the threshold is reached.")
new_profit_lst = add_insurance(prec_list,profit_list,threshold, payout, premium)
new_total_profit = sum(new_profit_lst)
st.markdown('New total profit: {}'.format(new_total_profit))
st.subheader("Precipitation and vendor profit with insurance over fair duration")
st.plotly_chart(plotly_prec_profit(prec_list, new_profit_lst))
