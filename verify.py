import pandas as pd
import decimal

def main():
    train = pd.DataFrame(pd.read_csv('train_class.csv'))
    validation = pd.DataFrame(pd.read_csv('validation_class.csv'))
    test = pd.DataFrame(pd.read_csv('test_class.csv'))
    print('total:')
    print(train.shape[0] + validation.shape[0] + test.shape[0])
    print('\n')
    print('train Yes')
    print(train.loc[train['Y'] == 'Yes'].shape[0])
    print('train No')
    print(train.loc[train['Y'] == 'No'].shape[0])

    print('\n')
    print('validation Yes')
    print(validation.loc[validation['Y'] == 'Yes'].shape[0])
    print('validation No')
    print(validation.loc[validation['Y'] == 'No'].shape[0])

    print('\n')
    print('test Yes')
    print(test.loc[test['Y'] == 'Yes'].shape[0])
    print('test No')
    print(test.loc[test['Y'] == 'No'].shape[0])


if __name__ == '__main__':
    main()