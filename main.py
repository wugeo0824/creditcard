import pandas as pd
import decimal
import numpy as np
import defines
import model
import preprocess

def main():
    #preprocess and clean data with data EDUCATION 0, 5, 6 and MARRIAGE 0
    D = defines.Data()
    D.NO_NOISY = False
    print('preprocessing with noise data')
    # preprocess.updateD(D)
    # preprocess.cleanData()
    # preprocess.statistics()
    # preprocess.createDummyVar()
    model.updateD(D)
    model.compareWithROC()

    # preprocess and clean data without data EDUCATION 0, 5, 6 and MARRIAGE 0

    D.NO_NOISY = True
    print('preprocessing without noise data')
    # preprocess.updateD(D)
    # preprocess.cleanData()
    # preprocess.statistics()
    # preprocess.createDummyVar()
    model.updateD(D)
    model.compareWithROC()

if __name__ == '__main__':
    main()