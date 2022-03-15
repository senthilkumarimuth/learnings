import pandas as pd
from sklearn import datasets

from evidently.dashboard import Dashboard
from evidently.dashboard import DataDriftTab

iris = datasets.load_iris()
iris_frame = pd.DataFrame(iris.data, columns = iris.feature_names)

iris_data_drift_report = Dashboard(tabs=[DataDriftTab()])
iris_data_drift_report.calculate(iris_frame[:100], iris_frame[100:], column_mapping = None)

iris_data_drift_report.show()

import textwrap

def wrap(string, max_width):
    temp_list = []
    flag = True
    str_list = list(string)
    print('here1')
    while flag == True:
        print('here')
        sli_str = ''.join(str_list[:max_width])
        str_list = str_list[max_width:]
        if len(str_list) ==0:
            flag = False
        temp_list.append(sli_str)
    result = "\n".join(temp_list)
    return "\n".join(temp_list)

if __name__ == '__main__':
    string, max_width = input(), int(input())
    result = wrap(string, max_width)
    print(result)
