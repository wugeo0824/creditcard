import pandas as pd
import decimal

def main():
    all = pd.read_csv('alldata.csv')
    df = pd.DataFrame(all)

    #check missing or NA
    #print(df.apply(lambda x: sum(x.isnull()), axis=0))

    total_male = df[df['X2'] == 1]
    total_female = df[df['X2'] == 2]

    #print(total_male['X2'].value_counts())

    #get gender counts
    print(df['X2'].value_counts())
    total_gender = df['X2'].value_counts().values
    total = total_gender[0] + total_gender[1]

    # get 40% for male,60% for female
    female_pct = round(total_gender[0] / total, 2)
    male_pct = round(total_gender[1] / total, 2)

    #split the dataset: 40% for train, 10% for validation, and 50% for test
    train_count = 0.4 * total
    validation_count = 0.1 * total
    test_count = 0.5 * total

    train_male_count = int(train_count * male_pct)
    validation_male_count = int(validation_count * male_pct)
    test_male_count = int(test_count * male_pct)

    train_female_count = int(train_count * female_pct)
    validation_female_count = int(validation_count * female_pct)
    test_female_count = int(test_count * female_pct)

    # print(train_male_count)
    # print(train_female_count)
    # print(validation_male_count)
    # print(validation_female_count)
    # print(test_male_count)
    # print(test_female_count)

    train_male_df = total_male.head(train_male_count)
    train_female_df = total_female.head(train_female_count)
    train = pd.concat([train_female_df, train_male_df])
    print(train)
    train.to_csv('train.csv', index=False)

    validation_male_df = total_male[train_male_count : train_male_count + validation_male_count]
    validation_female_df = total_female[train_female_count: train_female_count + validation_female_count]
    validation = pd.concat([validation_male_df, validation_female_df])
    print(validation)
    validation.to_csv('validation.csv', index=False)

    test_male_df = total_male[train_male_count + validation_male_count:]
    test_female_df = total_female[train_female_count + validation_female_count:]
    test = pd.concat([test_male_df, test_female_df])
    print(test)
    test.to_csv('test.csv', index=False)

if __name__ == '__main__':
    main()