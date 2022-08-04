from papers.meta_classes import Publication, Finding
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


class Jeong2021Math(Publication):
    DEFAULT_PAPER_ATTRIBUTES = {
        'id': 'jeong2021math',
        'length_pages': 12,
        'authors': ['Haewon Jeong', 'Michael D. Wu', 'Nilanjana Dasgupta', 'Muriel Médard', 'Flavio P. Calmon'],
        'journal': None,
        'year': 2021,
        'current_citations': None,
        'base_dataframe_pickle': None
    }
    DATAFRAME_COLUMNS = [
        "X1TXMSCR", "X1RACE", "X1MTHID", "X1MTHEFF", "X1MTHINT", "X1FAMINCOME", "X1HHNUMBER", "X1P1RELATION",
        "X1PAR1EMP", "X1PARRESP", "X1SCHOOLBEL", "X1STU30OCC2", "S1M8GRADE", "S1LANG1ST", "S1TEPOPULAR",
        "S1TEMAKEFUN", "S1MTHCOMP", "S1SCICOMP", "S1APCALC", "S1IBCALC", "S1MTCHVALUES", "S1MTCHINTRST",
        "S1MTCHFAIR", "S1MTCHRESPCT", "S1MTCHCONF", "S1MTCHEASY", "X1PAR1EDU", "X1PAR2EDU", "X1PAR1OCC2",
        "X1PAR2OCC2", "P1REPEATGRD", "P1ELLEVER", "P1MARSTAT", "P1YRBORN1", "P1YRBORN2", "P1JOBNOW1",
        "P1JOBONET1_STEM1", "P1JOBONET2_STEM1", "P1HHTIME", "P1EDUASPIRE", "P1EDUEXPECT", "P1MTHHWEFF",
        "P1SCIHWEFF", "P1ENGHWEFF", "P1MTHCOMP", "P1SCICOMP", "P1ENGCOMP", "P1MUSEUM", "P1COMPUTER",
        "P1FIXED", "P1SCIFAIR", "P1SCIPROJ", "P1STEMDISC", "P1NOACT", "P1CAMPMS", "P1CAMPOTH", "P1NOOUTSCH"
    ]
    RACE_GROUP_MAP = {
        8: 'WHITE_ASIAN',
        3: 'BLACK_HISPANIC_NATIVE',
        4: 'BLACK_HISPANIC_NATIVE',
        5: 'BLACK_HISPANIC_NATIVE',
        2: 'WHITE_ASIAN',
        6: 'MULTIRACIAL_OTHER',
        7: 'MULTIRACIAL_OTHER',
        1: 'BLACK_HISPANIC_NATIVE',
        -9: 'UNKNOWN',
    }
    RACE_CLASSES = ['WHITE_ASIAN', 'BLACK_HISPANIC_NATIVE']
    FILENAME = 'jeong2021math'

    def __init__(self, dataframe=None, filename=None):
        if filename is not None:
            dataframe = pd.read_pickle(filename)
        elif dataframe is not None:
            dataframe = dataframe
        else:
            dataframe = self._recreate_dataframe()
        super().__init__(dataframe)
        self.FINDINGS = self.FINDINGS + []

    @classmethod
    def preprocess(cls, data, n_neighbors=1):
        features = data.drop(columns=['X1TXMSCR', 'RACE_GROUP'])
        race_group = data['RACE_GROUP']
        target = data['X1TXMSCR']
        _features = features.apply(lambda col: pd.Series(
            LabelEncoder().fit_transform(col[col.notnull()]),
            index=col[col.notnull()].index
        ))
        _features = KNNImputer(n_neighbors=n_neighbors).fit_transform(_features)
        _features = MinMaxScaler().fit_transform(_features)
        processed_data = pd.DataFrame(_features, columns=features.columns)
        processed_data['target'] = np.where(target > target.median(), 1, 0)
        processed_data['RACE_GROUP'] = race_group.values
        return processed_data

    def _recreate_dataframe(self, filename='jeong2021math_dataframe.pickle'):
        # school_survey = pd.read_csv('../data/36423-0001-Data.tsv', sep='\t')
        student_survey = pd.read_csv('../data/math/36423-0002-Data.tsv', sep='\t')
        data = student_survey[self.DATAFRAME_COLUMNS]
        data['RACE_GROUP'] = data['X1RACE'].map(self.RACE_GROUP_MAP)
        data = data[data['RACE_GROUP'].isin(self.RACE_CLASSES)]
        data = data.dropna(subset=['X1TXMSCR'])  # target shouldnt be na
        for column_name in self.DATAFRAME_COLUMNS:
            data[column_name] = np.where(data[column_name] < -6, None, data[column_name])
        data = data.drop(columns=['X1RACE'])
        data = data[data.isna().sum(axis=1) < len(data.columns) / 2]  # drop with more than half na features
        cont_features = ["X1MTHID", "X1MTHEFF", "X1MTHINT", "X1FAMINCOME", "X1HHNUMBER", "X1SCHOOLBEL", "X1TXMSCR"]
        cat_features = data.columns[~data.columns.isin(cont_features)]
        data[cont_features] = data[cont_features].astype(np.float64)
        data[cat_features] = data[cat_features].astype('category')
        data = self.preprocess(data)
        # print(data.RACE_GROUP.value_counts())
        print(data.shape)
        data.to_pickle(filename)  # 10156 training set
        return data

    def train_classifier(self, X_train, y_train, random_state):
        model = RandomForestClassifier(max_depth=16, min_samples_leaf=3, n_estimators=100, random_state=random_state)
        model.fit(X_train, y_train)
        return model

    def evaluate_classifier(self, model, X_test, X_test_race_group, y_test):
        evaluation_df = pd.concat([X_test, X_test_race_group], axis=1)
        evaluation_df['target'] = y_test
        evaluation_df['y_prediction'] = model.predict(X_test)
        return evaluation_df

    def evaluate(self, data):
        scores_dataframes = []
        for random_state in range(30):
            X_train, X_train_race_group, y_train, X_test, X_test_race_group, y_test = \
                self.train_test_split(data, random_state)
            model = self.train_classifier(X_train, y_train, random_state)
            evaluation_df = self.evaluate_classifier(model, X_test, X_test_race_group, y_test)
            for class_name in self.RACE_CLASSES:
                seed_scores = self.calculate_scores(evaluation_df, class_name)
                scores_dataframes.append(seed_scores)
        return pd.DataFrame(scores_dataframes)

    def calculate_scores(self, df, class_name):
        y_true, y_pred = df[df.RACE_GROUP == class_name][['target', 'y_prediction']].T.to_numpy()
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return {
            'class_name': class_name,
            'SIZE': len(y_true),
            'PBR': (tp + fp) / (tp + fp + fn + tn),
            'FPR': fp / (fp + tn),
            'FNR': fn / (tp + fn),
            'ACC': (tp + tn) / (tp + fp + fn + tn)
        }

    def train_test_split(self, data, random_state=42, test_size=0.3):
        X = data.drop(columns=['target'])
        y = data['target']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        X_train_race_group = X_train.loc[:, 'RACE_GROUP']
        X_train = X_train.drop(columns=['RACE_GROUP'])
        X_test_race_group = X_test.loc[:, 'RACE_GROUP']
        X_test = X_test.drop(columns=['RACE_GROUP'])
        return X_train, X_train_race_group, y_train, X_test, X_test_race_group, y_test

    def finding_rf(self):
        data = self.dataframe
        result = self.evaluate(data)
        print(result.groupby('class_name').mean())
        print(result.groupby('class_name').std())


if __name__ == '__main__':
    paper = Jeong2021Math(filename = 'jeong2021math_dataframe.pickle')
    paper.finding_rf()
