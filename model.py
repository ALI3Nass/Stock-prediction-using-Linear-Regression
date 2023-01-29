import pandas as pd
import matplotlib.pyplot as plt
import pandas_ta 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

df = pd.read_csv('TSLA.csv')
df.set_index(pd.DatetimeIndex(df['Date']), inplace=True)
df = df[['Adj Close']]

df.ta.ema(close='Adj Close', length=10, append=True)
df = df.iloc[9:]

X_train, X_test, y_train, y_test = train_test_split(df[['Adj Close']], df[['EMA_10']], test_size=.1)

model = LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
# df_pred = pd.concat([y_test, pd.DataFrame(y_pred, columns=['Predicted'], index=y_test.index)], axis=1)

plt.plot(y_pred, label='Actual Data')
# plt.plot(df_pred['Predicted'], label='Prediction')
plt.xlabel('Index')
plt.ylabel('Price')
plt.ylim(bottom=0)
plt.grid(True)
plt.legend()
plt.title('Tesla Stock Price')
plt.show()

print("Model Coefficient:                 ", model.coef_)
print("\nMean Absolute Error:             ", mean_absolute_error(y_test, y_pred))
print("\nCoefficient of Determination:    ", r2_score(y_test, y_pred))
