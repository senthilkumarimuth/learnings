#import
import pandas as pd
from prophet import Prophet

# Python
df = pd.read_csv('https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv')
print(df.head())

#model building
model = Prophet()
model.fit(df)

#future dataset
future = model.make_future_dataframe(periods=365)
future.tail()

# predict
forecast = model.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

#plot
fig1 = model.plot(forecast)

# Python
fig2 = model.plot_components(forecast)


