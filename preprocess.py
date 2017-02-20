import pandas as pd

def main():
    train = pd.DataFrame(pd.read_csv('train.csv'))
    validation = pd.DataFrame(pd.read_csv('validation.csv'))
    test = pd.DataFrame(pd.read_csv('test.csv'))

    train['Y'] = train['Y'].map({1:'Yes', 0: 'No'})
    train.to_csv('train_class.csv', index=False)

    validation['Y'] = validation['Y'].map({1: 'Yes', 0: 'No'})
    validation.to_csv('validation_class.csv', index=False)

    test['Y'] = test['Y'].map({1:'Yes', 0: 'No'})
    test.to_csv('test_class.csv', index=False)


if __name__ == '__main__':
    main()