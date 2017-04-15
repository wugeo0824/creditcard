import pandas as pd
import defines

total_strange_default = 0
total_strange_ok = 0

D = defines.Data()

def updateD(data):
    global D
    D = data

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

    if row['BILL_AMT1'] == 0 and bill == 0:
        if row[D.ATTR] == 1:
            total_strange_default += 1
            flag = -2

        if row[D.ATTR] == 0:
            total_strange_ok += 1
            flag = -2

    return flag

def fixEducation(row):
    if row['EDUCATION'] > 4:
        return 4

    return row['EDUCATION']

def cleanData():
    df = pd.read_csv(D.SOURCE)

    df['PAY_POSSIBLE_WRONG'] = df.apply(checkPayStatus, axis=1)
    df['BILL_POSSIBLE_WRONG'] = df.apply(checkTotalBill, axis=1)

    df.drop(df[df.PAY_POSSIBLE_WRONG == -2].index, inplace=True)
    df.drop(df[df.BILL_POSSIBLE_WRONG == -2].index, inplace=True)

    df = df.drop(['PAY_POSSIBLE_WRONG', 'BILL_POSSIBLE_WRONG'], axis=1)

    if D.NO_NOISY:
        df['EDUCATION'] = df.apply(fixEducation, axis=1)
        df = df[df['MARRIAGE'] != 0]
        df = df[df['EDUCATION'] != 0]

    print(total_strange_default)
    print(total_strange_ok)

    df.to_csv(D.getFile(), index=False)

def readCleanData():
    return pd.read_csv(D.getFile())

def statistics():
    df = readCleanData()
    print('Total records: ')
    print(df[D.ATTR].count())
    print('Default sum: ')
    print(df[D.ATTR].sum())

def createDummyVar():
    df = readCleanData()
    df = pd.get_dummies(df, prefix='GENDER_', columns=['SEX'])
    df = pd.get_dummies(df, prefix='EDUCATION_', columns=['EDUCATION'])
    df = pd.get_dummies(df, prefix='MARRIAGE_', columns=['MARRIAGE'])

    df.to_csv(D.getDummyFile(), index=False)

# def convert(row, i):
#     j = str(i)
#     bill = row['BILL_AMT' + j]
#     pay = row['PAY_AMT' + j]
#     if bill <= 0:
#         return 0
#     else:
#         if pay == 0:
#             return bill
#         else:
#             return bill / pay
#     return 0

#no improvement
# def featureEngineering():
#     df = readDataWithDummy()
#     df['BILL_TO_PAY1'] = df.apply(lambda row: convert(row, 1), axis=1)
#     df['BILL_TO_PAY2'] = df.apply(lambda row: convert(row, 2), axis=1)
#     df['BILL_TO_PAY3'] = df.apply(lambda row: convert(row, 3), axis=1)
#     df['BILL_TO_PAY4'] = df.apply(lambda row: convert(row, 4), axis=1)
#     df['BILL_TO_PAY5'] = df.apply(lambda row: convert(row, 5), axis=1)
#     df['BILL_TO_PAY6'] = df.apply(lambda row: convert(row, 6), axis=1)
#
#     df.to_csv('all_data_clean_with_dummy_extra.csv', index=False)

def main():
    cleanData()
    statistics()
    createDummyVar()
    #featureEngineering()

if __name__ == '__main__':
    main()