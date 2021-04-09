data_model4_test['contact_predicted']=data_model4_test['avg_prob'].apply(lambda x: 1 if x>0.5 else 0)

# print two if two column values are equal else print false
df['col1'].equals(df['col2'])