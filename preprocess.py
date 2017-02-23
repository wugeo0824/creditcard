import pandas as pd
import decimal
from sklearn.utils import shuffle

def main():
    train = pd.DataFrame(pd.read_csv('train_class.csv'))
    train['Y'] = train['Y'].map({'Yes': 1, 'No': 0})
    train.to_csv('train.csv', index=False)

    validation = pd.DataFrame(pd.read_csv('validation_class.csv'))
    validation['Y'] = validation['Y'].map({'Yes': 1, 'No': 0})
    validation.to_csv('validation.csv', index=False)

    test = pd.DataFrame(pd.read_csv('test_class.csv'))
    test['Y'] = test['Y'].map({'Yes': 1, 'No': 0})
    test.to_csv('test.csv', index=False)
#
# def main():
#     all = pd.DataFrame(pd.read_csv('alldata.csv'))
#     all['Y'] = all['Y'].map({1: 'Yes', 0: 'No'})
#     total = all.shape[0]
#     #all.to_csv('alldata_class.csv', index=False)
#
#     full = pd.DataFrame(pd.read_csv('alldata_class.csv'))
#
#     default_data = full.loc[full['Y'] == 'Yes']
#     nondefault_data = full.loc[full['Y'] == 'No']
#
#     default_pct = round((default_data['Y'].value_counts()) / total, 2)
#     nondefault_pct = 1 - default_pct
#
#     print(default_pct)
#     print(nondefault_pct)
#
#     #40% training, 10% validation, 50% test
#     train_default_count =  int(0.4 * default_pct * total)
#     train_nondefault_count = int(0.4 * nondefault_pct * total)
#
#     validation_default_count = int (0.1 * default_pct * total)
#     validation_nondefault_count = int(0.1  * nondefault_pct * total)
#
#     train_default_df = default_data.head(train_default_count)
#     train_nondefault_df = nondefault_data.head(train_nondefault_count)
#     train = pd.concat([train_default_df, train_nondefault_df])
#     #print(train)
#     train = shuffle(train)
#     train.to_csv('train_class.csv', index=False)
#
#     validation_default_df = default_data[train_default_count: train_default_count + validation_default_count]
#     validation_nondefault_df = nondefault_data[train_nondefault_count: train_nondefault_count + validation_nondefault_count]
#     validation = pd.concat([validation_default_df, validation_nondefault_df])
#     # print(validation)
#     validation = shuffle(validation)
#     validation.to_csv('validation_class.csv', index=False)
#
#     test_default_df = default_data[train_default_count + validation_default_count:]
#     test_nondefault_df = nondefault_data[train_nondefault_count + validation_nondefault_count:]
#     test = pd.concat([test_default_df, test_nondefault_df])
#     # print(test)
#     test = shuffle(test)
#     test.to_csv('test_class.csv', index=False)


if __name__ == '__main__':
    main()