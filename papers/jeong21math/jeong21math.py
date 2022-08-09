from papers.meta_classes import Publication, Finding
from papers.meta_classes import NonReproducibleFindingException
from papers.file_utils import PathSearcher
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
        'base_dataframe_pickle': 'jeong2021math_dataframe.pickle'
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
    RANDOM_STATE_MAX = 30

    def __init__(self, dataframe=None, filename=None, path=None):
        if dataframe is None:
            if path is None:
                path = self.DEFAULT_PAPER_ATTRIBUTES['id']
            self.path_searcher = PathSearcher(path)
            if filename is None:
                filename = self.DEFAULT_PAPER_ATTRIBUTES['base_dataframe_pickle']
            try:
                dataframe = pd.read_pickle(self.path_searcher.get_path(filename))
            except FileNotFoundError:
                dataframe = self._recreate_dataframe()
        super().__init__(dataframe)
        self.train_results, self.test_results = self.score_by_class(self.dataframe)
        self.table_1 = self.get_results_by_class(self.test_results)
        self.FINDINGS = self.FINDINGS + [
            Finding(self.finding_2_1),
            Finding(self.finding_3_1),
            Finding(self.finding_3_2),
            Finding(self.finding_3_3),
            Finding(self.finding_3_4),
            Finding(self.finding_3_5),
            # Finding(self.finding_3_6),
            Finding(self.finding_3_7),
            Finding(self.finding_3_8),
            # Finding(self.finding_4_1),
            # Finding(self.finding_4_2)
        ]

    def _recreate_dataframe(self, filename='jeong2021math_dataframe.pickle'):
        # school_survey = pd.read_csv('data/36423-0001-Data.tsv', sep='\t')
        student_survey = pd.read_csv('data/36423-0002-Data.tsv', sep='\t')
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

    @staticmethod
    def preprocess(data, n_neighbors=3):
        features = data.drop(columns=['X1TXMSCR', 'RACE_GROUP'])
        race_group = data['RACE_GROUP']
        target = data['X1TXMSCR']
        _features = features.apply(
            lambda col: pd.Series(
                LabelEncoder().fit_transform(
                    col[col.notnull()]
                ), index=col[col.notnull()].index)
        )
        _features = KNNImputer(n_neighbors=n_neighbors).fit_transform(_features)
        _features = MinMaxScaler().fit_transform(_features)
        processed_data = pd.DataFrame(_features, columns=features.columns)
        processed_data['target'] = np.where(target > target.median(), 1, 0)
        processed_data['RACE_GROUP'] = race_group.values
        return processed_data

    @staticmethod
    def train_classifier(X_train, y_train, random_state):
        model = RandomForestClassifier(
            max_depth=16, min_samples_leaf=3, n_estimators=100, random_state=random_state,
        )
        model.fit(X_train, y_train)
        return model

    @staticmethod
    def evaluate_classifier(model, X_train, X_train_race_group, y_train, X_test, X_test_race_group, y_test):
        train_df = pd.concat([X_train, X_train_race_group], axis=1)
        train_df['target'] = y_train
        train_df['y_prediction'] = model.predict(X_train)
        evaluation_df = pd.concat([X_test, X_test_race_group], axis=1)
        evaluation_df['target'] = y_test
        evaluation_df['y_prediction'] = model.predict(X_test)
        return train_df, evaluation_df

    @classmethod
    def score(cls, data, model_type=None, random_states_num=None):
        if model_type is not None and model_type == 'dummy':
            X = data[['S1M8GRADE']]
        else:
            X = data.drop(columns=['target', 'RACE_GROUP'])
        y = data['target']
        scores = []
        for random_state in range(random_states_num or cls.RANDOM_STATE_MAX):
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)
            model = cls.train_classifier(X_train, y_train, random_state)
            scores.append(model.score(X_test, y_test))
        return np.mean(scores), np.std(scores)

    @classmethod
    def score_by_class(cls, data, test_size=0.3, random_states_num=None):
        train_scores_dataframes = []
        scores_dataframes = []
        for random_state in range(random_states_num or cls.RANDOM_STATE_MAX):
            X_train, X_train_race_group, y_train, X_test, X_test_race_group, y_test = \
                cls.train_test_split(data, random_state, test_size)
            model = cls.train_classifier(X_train, y_train, random_state)
            train_df, evaluation_df = cls.evaluate_classifier(
                model, X_train, X_train_race_group, y_train, X_test, X_test_race_group, y_test
            )
            for class_name in cls.RACE_CLASSES:
                train_seed_scores = cls.calculate_scores(train_df, class_name)
                train_scores_dataframes.append(train_seed_scores)
                seed_scores = cls.calculate_scores(evaluation_df, class_name)
                scores_dataframes.append(seed_scores)
        return pd.DataFrame(train_scores_dataframes), pd.DataFrame(scores_dataframes)

    @staticmethod
    def calculate_scores(df, class_name):
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

    @staticmethod
    def train_test_split(data, random_state=42, test_size=0.3):
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

    @staticmethod
    def get_results_by_class(result):
        return result.groupby('class_name').mean(), result.groupby('class_name').std()

    def finding_2_1(self):
        """
        With the HSLS dataset, we observe that prediction accuracy is 68.2 ± 0.1 % if we predict students’ math
        performance in the 9th grade based only on their past performance. Accuracy improves to 75.0 ± 0.1 %,
        by utilizing more features such as students’ and parents’ survey responses.
        """
        data = self.dataframe
        single_feature_model_acc, single_feature_model_std = self.score(data, 'dummy')
        all_feature_model_acc, all_feature_model_std = self.score(data)
        single_feature_model_acc *= 100
        single_feature_model_std *= 100
        all_feature_model_acc *= 100
        all_feature_model_std *= 100
        all_findings = [single_feature_model_acc, single_feature_model_std, all_feature_model_acc, all_feature_model_std]
        hard_findings = [np.allclose(single_feature_model_acc, 68.2, atol=10e-2),
                         np.allclose(single_feature_model_std, 0.1, atol=10e-2),
                         np.allclose(all_feature_model_acc, 75.0, atol=10e-2),
                         np.allclose(all_feature_model_std, 0.1, atol=10e-2)]
        return all_findings, [], hard_findings

    def finding_3_1(self):
        """
        First, notice that the difference in accuracy between WA and BHN is negligible.
        """

        table_1_mean, table_1_std = self.table_1
        wa_acc = table_1_mean.loc['WHITE_ASIAN', 'ACC'] * 100
        bhn_acc = table_1_mean.loc['BLACK_HISPANIC_NATIVE', 'ACC'] * 100
        all_findings = [wa_acc, bhn_acc]
        hard_findings = [np.allclose(wa_acc, bhn_acc, atol=10e-1)]
        return all_findings, [], hard_findings

    def finding_3_2(self):
        """
        However, FNR was considerably smaller for WA students compared to BHN. The relative difference in FNR was up
        to 78%. This implies that WA students are less prone to get an underestimated prediction by the ML model.
        """
        wa_fnr = self.test_results[self.test_results['class_name'] == 'WHITE_ASIAN']['FNR'].values * 100
        bhn_fnr = self.test_results[self.test_results['class_name'] == 'BLACK_HISPANIC_NATIVE']['FNR'].values * 100
        diff_fnr = np.abs(wa_fnr - bhn_fnr)
        soft_findings = [np.alltrue(wa_fnr < bhn_fnr)]
        hard_findings = [np.allclose(np.max(diff_fnr), 78, atol=10e-1)]
        return diff_fnr, soft_findings, hard_findings

    def finding_3_3(self):
        """
        At the same time, FPR is 42% higher for WA than BHN students. In other words, WA students more frequently
        receive the benefit of the doubt from the trained ML model.
        """
        wa_fpr = self.test_results[self.test_results['class_name'] == 'WHITE_ASIAN']['FPR'].values * 100
        bhn_fpr = self.test_results[self.test_results['class_name'] == 'BLACK_HISPANIC_NATIVE']['FPR'].values * 100
        diff_fpr = np.abs(wa_fpr - bhn_fpr)
        soft_findings = [np.alltrue(wa_fpr > bhn_fpr)]
        hard_findings = [np.allclose(np.max(diff_fpr), 42, atol=10e-1)]
        return diff_fpr, soft_findings, hard_findings

    def finding_3_4(self):
        """
        We also observe that PBR is higher in WA students than in BHN students. This may reflect the difference in the
        ground truth data. The observed base rate was 0.57 for WA and 0.38 for BHN students. (difference = 0.19).
        """
        table_1_mean, table_1_std = self.table_1
        wa_pbr = table_1_mean.loc['WHITE_ASIAN', 'PBR'] * 100
        bhn_pbr = table_1_mean.loc['BLACK_HISPANIC_NATIVE', 'PBR'] * 100
        diff_pbr = np.abs(wa_pbr - bhn_pbr)
        all_findings = [wa_pbr, bhn_pbr, diff_pbr]
        soft_findings = [wa_pbr > bhn_pbr]
        hard_findings = [np.allclose(wa_pbr, 57,  atol=10e-1), np.allclose(bhn_pbr, 38,  atol=10e-1),
                         np.allclose(diff_pbr, 19,  atol=10e-1)]
        return all_findings, soft_findings, hard_findings

    def finding_3_5(self):
        """
        However, the PBR difference from the trained random forest models was about 0.23, indicating that the existing
        racial performance gap is exaggerated in the ML model’s predictions.
        """
        train_mean, train_1_std = self.get_results_by_class(self.train_results)
        wa_pbr = train_mean.loc['WHITE_ASIAN', 'PBR'] * 100
        bhn_pbr = train_mean.loc['BLACK_HISPANIC_NATIVE', 'PBR'] * 100
        diff_pbr = np.abs(wa_pbr - bhn_pbr)
        all_findings = [wa_pbr, bhn_pbr, diff_pbr]
        soft_findings = [wa_pbr > bhn_pbr]
        hard_findings = [np.allclose(diff_pbr, 23,  atol=10e-1)]
        return all_findings, soft_findings, hard_findings

    def finding_3_6(self):
        """
        By examining FPR and FNR, we discover that WA students are consistently given more benefit of the doubt, while
        BHN students are consistently underestimated in predicting their future math performance despite similar
        accuracy numbers for both groups. This shows that narrowly focusing on accuracy can give an illusion of fairness
        when there is significant discriminatory impact on students from historically underrepresented groups.
        Note: Conclusion from previous findings - SKIPPED.
        """
        return NotImplemented

    def finding_3_7(self):
        """
        The FPR of 0.30 for WA students (see Table 1) means that 30% of the students who would not perform well in the
        9th grade will be placed in the advanced class. They are given the benefit of the doubt and the opportunity to
        learn more advanced math. On the other hand, only 18% of the BHN students get the same benefit of the doubt (FPR=0.18).
        """
        table_1_mean, table_1_std = self.table_1
        wa_fpr = table_1_mean.loc['WHITE_ASIAN', 'FPR'] * 100
        bhn_fpr = table_1_mean.loc['BLACK_HISPANIC_NATIVE', 'FPR'] * 100
        all_findings = [wa_fpr, bhn_fpr]
        hard_findings = [np.allclose(wa_fpr, 30, atol=10e-1), np.allclose(bhn_fpr, 18, atol=10e-1)]
        return all_findings, [], hard_findings

    def finding_3_8(self):
        """
        The FNR of 0.21 in WA students indicates that 21% ofWA students who would in fact perform well in the future
        will be placed in the basic class by the ML algorithm. For BHN students, a startlingly high 37% will be
        incorrectly placed in the basic class, their academic potential ignored by the algorithm.
        """
        table_1_mean, table_1_std = self.table_1
        wa_fnr = table_1_mean.loc['WHITE_ASIAN', 'FNR'] * 100
        bhn_fnr = table_1_mean.loc['BLACK_HISPANIC_NATIVE', 'FNR'] * 100
        all_findings= [wa_fnr, bhn_fnr]
        hard_findings = [np.allclose(wa_fnr, 21, atol=10e-1), np.allclose(bhn_fnr, 37, atol=10e-1)]
        return all_findings, [], hard_findings

    def finding_4_1(self):
        """
        Focusing solely on accuracy may lead to the incorrect conclusion that the effect of different racial compositions
        of a training set is minute: the accuracy for each group does not vary more than 0.05 as we change p from 0 to 1
        (i.e., 0% to 100% BHN).
        Note: Authors state conflicting statement about the train test split for this result
        """
        raise NonReproducibleFindingException

    def finding_4_2(self):
        """
        However, FPR and FNR metrics change drastically with different racial compositions of the training set. FPR
        monotonically decreases and FNR monotonically increases for both BHN and WA students as we increase p from 0%
        BHN to 100% BHN. The range of FPR difference is from ∼0.4 to 0.1 and FNR moves from 0.2 to 0.5. The gaps in FPR
        and FNR remain throughout different values of p, but they reduce slightly around p = 0.
        Note: Authors state conflicting statement about the train test split for this result
        """
        raise NonReproducibleFindingException


if __name__ == '__main__':
    paper = Jeong2021Math()
    for find in paper.FINDINGS:
        print(find.run())
