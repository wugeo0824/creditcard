import pandas as pd
import decimal
from sklearn.utils import shuffle

total_strange_default = 0
total_strange_ok = 0
attr = 'DEFAULT_PAYMENT_NEXT_MONTH'

def checkPayStatus(row):
    flag = -2

    if row['PAY_0'] == -2:
        return flag

    for i in range(2, 7):
        if row['PAY_' + str(i)] == -2:
            return flag

    return 0

def checkTotalBill(row):
    global total_strange_default
    global total_strange_ok

    flag = 0
    bill = 0

    for i in range(1, 7):
        bill += float(row['BILL_AMT' + str(i)])

    if row['BILL_AMT1'] == 0 and bill == 0 and row[attr] == 1:
        total_strange_default += 1
        flag = -2

    if row['BILL_AMT1'] == 0 and bill == 0 and row[attr] == 0:
        total_strange_ok += 1
        flag = -2

    return flag

def cleanData():
    df = pd.read_csv('credit_cards_default.csv')

    df['PAY_WRONG'] = df.apply(checkPayStatus, axis=1)
    df['BILL_WRONG'] = df.apply(checkTotalBill, axis=1)

    df.drop(df[df.PAY_WRONG == -2].index, inplace=True)
    df.drop(df[df.BILL_WRONG == -2].index, inplace=True)

    df = df.drop(['PAY_WRONG', 'BILL_WRONG'], axis=1)

    print(total_strange_default)
    print(total_strange_ok)

    df.to_csv('all_data_clean.csv', index=False)

def readCleanData():
    return pd.read_csv('all_data_clean.csv')

def statistics():
    df = readCleanData()
    print(df.describe())
    print('Default sum: ')
    print(df[attr].sum())

def createDummyVar():
    df = readCleanData()
    df = pd.get_dummies(df, prefix='GENDER_', columns=['SEX'])
    df = pd.get_dummies(df, prefix='EDUCATION_', columns=['EDUCATION'])
    df = pd.get_dummies(df, prefix='MARRIAGE_', columns=['MARRIAGE'])
    df.to_csv('all_data_clean_with_dummy.csv', index=False)


def convert(row):

    for i in range(1, 7):
        j = str(i)
        bill = row['BILL_AMT' + j]
        pay = row['PAY_AMT' + j]
        if bill <= 0:
            row['BILL_TO_PAY' + j] = 0
        else:
            if pay == 0:
                row['BILL_TO_PAY' + j] = 1
            else:
                row['BILL_TO_PAY' + j] = bill / pay


def featureEngineering():
    df = readDataWithDummy()
    df = df.apply(convert, axis=1)
    # for index, row in df.iterrows():
    #     row = convert(row)

    df.to_csv('all_data_clean_with_dummy_extra.csv', index=False)

def readDataWithDummy():
    return pd.read_csv('all_data_clean_with_dummy.csv')

def main():
    #cleanData()
    #statistics()
    #createDummyVar()
    featureEngineering()




if __name__ == '__main__':
    main()