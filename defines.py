class Data:
    def __init__(self):
        self.ATTR = 'DEFAULT_PAYMENT_NEXT_MONTH'
        self.SOURCE = 'credit_cards_default.csv'
        self.SOURCE_CLEAN_1 = 'all_clean_no_noise.csv'
        self.SOURCE_CLEAN_2 = 'all_clean_has_noise.csv'
        self.SOURCE_DUMMY_1 = 'all_clean_dummy_no_noise.csv'
        self.SOURCE_DUMMY_2 = 'all_clean_dummy_has_noise.csv'
        self.NO_NOISY = True
        self.PCT = 0.1

    def getFile(self):
        return self.SOURCE_CLEAN_1 if self.NO_NOISY else self.SOURCE_CLEAN_2

    def getDummyFile(self):
        return self.SOURCE_DUMMY_1 if self.NO_NOISY else self.SOURCE_DUMMY_2
