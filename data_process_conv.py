import re
import numpy as np
import pandas as pd
import joblib
import gc
from time import time
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

__import__('warnings').filterwarnings("ignore")


def convert_seconds(time_apply):
    # print(type(time_apply), time_apply)
    try:
        time_apply = float(time_apply)
    except ValueError:
        time_apply = 0
    if isinstance(time_apply, (int, float)):
        hrs = time_apply // 3600
        mns = time_apply % 3600
        sec = mns % 60
        time_string = ''
        if hrs:
            time_string = f'{hrs:.0f} час '
        if mns // 60 or hrs:
            time_string += f'{mns // 60:.0f} мин '
        return f'{time_string}{sec:.1f} сек'


def print_time(time_start, title=''):
    """
    Печать времени выполнения процесса
    :param time_start: время запуска в формате time.time()
    :param title: заголовок для сообщения
    :return:
    """
    title = f'{title} --> ' if title else ''
    time_apply = time() - time_start
    print(f'{title} Время обработки: {convert_seconds(time_apply)}'.strip())


def print_msg(msg):
    print(msg)
    return time()


def memory_compression(df, use_category=True, use_float=True, exclude_columns=None):
    """
    Изменение типов данных для экономии памяти
    :param df: исходный ДФ
    :param use_category: преобразовывать строки в категорию
    :param use_float: преобразовывать float в пониженную размерность
    :param exclude_columns: список колонок, которые нужно исключить из обработки
    :return: сжатый ДФ
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        # print(f'{col} тип: {tmp[col].dtype}', str(tmp[col].dtype)[:4])

        if exclude_columns and col in exclude_columns:
            continue

        if str(df[col].dtype)[:4] in 'datetime':
            continue

        elif str(df[col].dtype) not in ('object', 'category'):
            col_min = df[col].min()
            col_max = df[col].max()
            if str(df[col].dtype)[:3] == 'int':
                if col_min > np.iinfo(np.int8).min and \
                        col_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif col_min > np.iinfo(np.int16).min and \
                        col_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif col_min > np.iinfo(np.int32).min and \
                        col_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif col_min > np.iinfo(np.int64).min and \
                        col_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            elif use_float and str(df[col].dtype)[:5] == 'float':
                if col_min > np.finfo(np.float16).min and \
                        col_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif col_min > np.finfo(np.float32).min and \
                        col_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

        elif use_category and str(df[col].dtype) == 'object':
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print(f'Исходный размер датасета в памяти '
          f'равен {round(start_mem, 2)} мб.')
    print(f'Конечный размер датасета в памяти '
          f'равен {round(end_mem, 2)} мб.')
    print(f'Экономия памяти = {(1 - end_mem / start_mem):.1%}')
    return df


class DataTransform:
    def __init__(self, use_catboost=True, numeric_columns=None, category_columns=None,
                 drop_first=False, scaler=None, args_scaler=None, **kwargs):
        """
        Преобразование данных
        :param use_catboost: данные готовятся для catboost
        :param numeric_columns: цифровые колонки
        :param category_columns: категориальные колонки
        :param drop_first: из dummy переменных удалить первую колонку
        :param scaler: какой скейлер будем использовать
        :param degree: аргументы для скейлера, например: степень для полином.преобразования
        """
        self.use_catboost = use_catboost
        self.category_columns = [] if category_columns is None else category_columns
        self.numeric_columns = [] if numeric_columns is None else numeric_columns
        self.model_columns = []
        self.text_columns = []
        self.drop_duplicates = False
        self.drop_first = drop_first
        self.exclude_columns = []
        self.new_columns = []
        self.comment = {'drop_first': drop_first}
        self.transform_columns = None
        self.scaler = scaler
        self.args_scaler = args_scaler
        # Тип Векторайзера
        self.vectorizer = None
        # максимальная последовательность для векторизации
        self.vector_limit = None
        # диапазон N-грамм
        self.ngram_range = (1, 1)
        # Минимальное количество последовательностей для векторизации
        self.min_df = 1
        # Максимальное количество признаков
        self.max_features = None
        # обученный Векторайзер
        self.bigram_vectorizer = None
        # Тип Векторайзера content_category
        self.cont_cat_vectorizer = None
        # максимальная последовательность для векторизации content_category
        self.cont_cat_vector_limit = None
        # диапазон N-грамм content_category
        self.cont_cat_ngram_range = (1, 1)
        # Минимальное количество последовательностей для векторизации content_category
        self.cont_cat_min_df = 1
        # Максимальное количество признаков content_category
        self.cont_cat_max_features = None
        # обученный Векторайзер content_category
        self.cont_cat_bigram_vectorizer = None
        # удалять из трейна юзеров с пропусками
        self.drop_nan_users = False
        # удалять из трейна юзеров с пропусками только с X записями
        self.drop_nan_users_with_records = 0  # ставим количество записей
        # дополнительный ДФ из third_party
        self.conv_info = None
        # максимальный предел пропусков для нахождения моды и среднего
        self.max_limit_nans = 0.2
        # Объединить поля ssp и full_placement_id2
        self.concat_ssp_full_placement_id2 = False
        # Объединить поля user_id и bid_ip
        self.concat_user_id_bid_ip = False
        # Использовать укрупненные категории content_category
        self.use_category_groups = False
        # Создавать бинарные признаки из укрупненных категорий content_category
        self.content_category_groups = False
        # Создавать X признаков с разницей в секундах между текущим действием и предыдущими
        self.time_shift_counts = 0
        # Создавать бинарные признаки об информации по предыдущему действию
        self.time_shift_counts_binary = 0

    def cat_dummies(self, df):
        """
        Отметка категориальных колонок --> str для catboost
        OneHotEncoder для остальных
        :param df: ДФ
        :return: ДФ с фичами
        """
        # если нет цифровых колонок --> заполним их
        if self.category_columns and not self.numeric_columns:
            self.numeric_columns = [col_name for col_name in df.columns
                                    if col_name not in self.category_columns]
        # если нет категориальных колонок --> заполним их
        if self.numeric_columns and not self.category_columns:
            self.category_columns = [col_name for col_name in df.columns
                                     if col_name not in self.numeric_columns]

        for col_name in self.category_columns:
            if col_name in df.columns:
                if self.use_catboost:
                    df[col_name] = df[col_name].astype(str)
                else:
                    print(f'Трансформирую колонку: {col_name}')
                    # Create dummy variables
                    df = pd.get_dummies(df, columns=[col_name], drop_first=self.drop_first)

                    self.new_columns.extend([col for col in df.columns
                                             if col.startswith(col_name)])
        return df

    def apply_scaler(self, df):
        """
        Масштабирование цифровых колонок
        :param df: исходный ДФ
        :return: нормализованный ДФ
        """
        if not self.transform_columns:
            self.transform_columns = self.numeric_columns

        no_transform_columns = ('user_id', 'target', 'time', 'report')

        self.transform_columns = [col for col in self.transform_columns
                                  if col in df.columns and col not in no_transform_columns]

        print('self.transform_columns:', self.transform_columns)

        if self.scaler and self.transform_columns:
            print(f'Применяю scaler: {self.scaler.__name__} '
                  f'с аргументами: {self.args_scaler}')
            args = self.args_scaler if self.args_scaler else tuple()
            scaler = self.scaler(*args)
            scaled_data = scaler.fit_transform(df[self.transform_columns])
            if scaled_data.shape[1] != len(self.transform_columns):
                print(f'scaler породил: {scaled_data.shape[1]} колонок')
                new_columns = [f'pnf_{n:02}' for n in range(scaled_data.shape[1])]
                df = pd.concat([df, pd.DataFrame(scaled_data, columns=new_columns)], axis=1)
                self.exclude_columns.extend(self.transform_columns)
            else:
                df[self.transform_columns] = scaled_data

            self.comment.update(scaler=self.scaler.__name__, args_scaler=self.args_scaler)
        return df

    def fillna_fields(self, df, fields, numeric_fields=False):
        """
        Заполнение пропусков
        :param df: ДФ
        :param fields: список колонок для заполнения
        :param numeric_fields: поля имеют цифровой тип
        :return:
        """
        # определение количества пропусков в каждой колонке
        missing_counts = df[fields].isnull().sum()
        # проход по каждой колонке и заполнение пропущенных значений
        for field in fields:
            missing_count = missing_counts[field]
            if missing_count < self.max_limit_nans * len(df):
                if numeric_fields:
                    try:
                        df[field].fillna(df[field].mean(), inplace=True)
                    except:
                        df[field].fillna(-127, inplace=True)
                else:
                    try:
                        df[field].fillna(df[field].mode()[0], inplace=True)
                    except:
                        df[field].fillna('-127', inplace=True)
            else:
                if numeric_fields:
                    df[field].fillna(-127, inplace=True)
                else:
                    df[field].fillna('-127', inplace=True)
        return df

    @staticmethod
    def make_shift_times(row, idx):
        time_end = row['time']
        times = [x for x in row['grp_times'] if x < time_end]
        return (time_end - times[-idx]).total_seconds() if idx <= len(times) else -999

    def initial_preparation(self, df, file_df=None):
        """
        Общая первоначальная подготовка данных
        :param df: исходный ФД
        :param file_df: Предобработанный Файл .pkl с полным путём
        :return: обработанные ДФ
        """
        start_time = print_msg('Предобработка данных...')

        df['time'] = pd.to_datetime(df['time'])
        df['date'] = pd.to_datetime(df['time']).dt.date
        # df['number_weekday'] = df['time'].dt.weekday
        # df['number_day'] = df['time'].dt.day
        df['number_hour'] = df['time'].dt.hour

        # удаление пробелов
        df['landing_page'] = df['landing_page'].str.replace(r'\s+', '', regex=True)
        df['landing_page'] = df['landing_page'].str.replace(u'32', u'')

        df['landing_catalog'] = df['landing_page'].str.replace('&utm_term=.*', '',
                                                               regex=True,
                                                               flags=re.DOTALL)

        df['creative_size_height'] = df['creative_size'].apply(lambda x: int(x.split('x')[0]))
        df['creative_size_width'] = df['creative_size'].apply(lambda x: int(x.split('x')[1]))

        df['full_placement_id2'] = df['full_placement_id'].apply(lambda x: x.split('_')[0])

        df['device_screen'] = df['device_screen'].fillna('-1')

        df['host'] = df['landing_page'].apply(lambda x: x.split('/')[2])

        df[f'battr'] = df[f'battr'].fillna('-1')
        for i in [-1, 1, 2, 5, 6, 7, 15, 16]:
            df[f'battr_{i}'] = df['battr'].apply(lambda x: i if str(i) in x.split(',') else 0)

        ####
        for u in ['39804', '7802', '7952', '7974', '44874', '8035']:
            df.loc[df['user_segments'] == df['user_segments'], f'user_segments_{u}'] = (
                df.loc[df['user_segments'] == df['user_segments'], 'user_segments'].apply(
                    lambda x: 1 if u in x else 0).fillna(0).astype(int)
            )
        ####

        # время, когда мы последний раз считывали куки пользователя
        df['ud_cookie_ts'] = pd.to_datetime(df['ud_cookie_ts'])
        # вычисление разницы в секундах
        df['time_diff'] = (df['time'] - df['ud_cookie_ts']).dt.total_seconds()
        df['time_diff'] = df['time_diff'].fillna(-999).astype(int)

        # разделение значений и преобразование их в бинарные метки content_tags
        df['content_tags'].fillna('', inplace=True)
        df[['ct72', 'ct73']] = df['content_tags'].str.get_dummies(',')
        df[['ct72', 'ct73']] = df[['ct72', 'ct73']].astype(int)

        # Отметка, что 'user_id' известен или нет
        df['known'] = df['user_id'].map(lambda x: 0 if pd.isna(x) else 1)
        # Заполнение пропусков в 'user_id' значениями из 'bid_ip'
        df['user_id'].fillna(df['bid_ip'], inplace=True)

        # пытаемся заполнить пропуски в user_id из столбцов, по которым можно определить юзера
        fields = ['bid_ip', 'carrier_id', 'model', 'region_code', 'city', 'zip_code',
                  'ua_device_type', 'ua_os', 'ua_browser', 'ua_os_version',
                  'ua_browser_version']
        df['bid_ip'].fillna('-127', inplace=True)

        # заполнение пропусков
        df = self.fillna_fields(df, fields)

        # Создаем колонку 'unique_id' путем сцепления значений из колонок списка fields
        df['unique_id'] = df[fields].apply(lambda row: hash('_'.join(row.values.astype(str))),
                                           axis=1).astype(str)
        df['user_id'].fillna(df['unique_id'], inplace=True)

        # Добавляем колонку 'record_count' с количеством записей для каждого 'user_id'
        df['record_count'] = df.groupby('user_id')['bid_ip'].transform('count').fillna(0)

        # Добавляем колонку 'record_catalog' с количеством записей для каждого grp_cols
        grp_cols = ['user_id', 'landing_catalog']
        df['record_catalog'] = df.groupby(grp_cols)['bid_ip'].transform('count').fillna(0)

        df['historical_viewability'] = df['historical_viewability'].fillna(-111).astype(int)

        print('\nОбработка колонки user_segments\n')
        tqdm.pandas()
        df.user_segments = df.user_segments.progress_apply(
            lambda x: '' if pd.isna(x) else ' '.join(sorted(x.replace(' ', '').split(','),
                                                            key=int)))
        print('\nОбработка колонки content_category\n')
        df.content_category = df.content_category.progress_apply(
            lambda x: '' if pd.isna(x) else ' '.join(sorted(x.replace(' ', '').split(','))))

        new_cat_features = ['ads_txt_support', 'gdpr_regulation', 'user_fraud_state',
                            'creative_id', 'user_detection_type', 'ua_type', 'user_status']
        # изменение типа данных на str
        df[new_cat_features] = df[new_cat_features].astype(str)

        # заполнение пропусков
        df = self.fillna_fields(df, new_cat_features + ['timezone_offset'])
        df.timezone_offset = (df.timezone_offset / 60).astype(int)

        if hasattr(self, 'concat_ssp_full_placement_id2'):
            if self.concat_ssp_full_placement_id2:
                cols = ['ssp', 'full_placement_id2']
                df['ssp_full_placement_id2'] = df[cols].apply(
                    lambda row: '_'.join(row.values.astype(str)), axis=1)
                self.exclude_columns.extend(cols)

        if hasattr(self, 'concat_user_id_bid_ip') and self.concat_user_id_bid_ip:
            cols = ['user_id', 'bid_ip']
            df['user_id'] = df[cols].apply(
                lambda row: '_'.join(row.values.astype(str)), axis=1)
            self.exclude_columns.append('bid_ip')

        if hasattr(self, 'use_category_groups') and self.use_category_groups:
            df['content_groups'] = df.content_category.progress_apply(
                lambda x: set() if pd.isna(x) else set([s.split('-')[0] for s in x.split()]))

            df.content_category = df.content_groups.progress_apply(
                lambda x: ' '.join(sorted(x)))

            df.drop('content_groups', axis=1, inplace=True)
            self.comment['use_category_groups'] = True

        if hasattr(self, 'content_category_groups') and self.content_category_groups:
            df['content_groups'] = df.content_category.progress_apply(
                lambda x: set() if pd.isna(x) else set([s.split('-')[0] for s in x.split()]))

            # '4hdkao6q' - это не учитываем
            for idx in tqdm(range(26), total=26):
                cat = f'iab{idx + 1}'
                df[f'cg_{cat}'] = df['content_groups'].map(lambda x: cat in x).astype(int)

            df.drop('content_groups', axis=1, inplace=True)
            self.exclude_columns.append('content_category')
            self.comment['content_category_groups'] = True

        if hasattr(self, 'time_shift_counts') and self.time_shift_counts:
            grp_cols = ['user_id', 'tag_id']
            print(f'Группировка по {grp_cols} ...')
            grp = df.groupby(grp_cols, as_index=False).agg(grp_times=('time', list))
            # Сортировка времени
            tqdm.pandas()
            grp.grp_times = grp.grp_times.progress_apply(lambda x: sorted(x))
            # добавление группировки в ДФ
            df = df.merge(grp, on=grp_cols, how='left')
            df['grp_times'] = df['grp_times'].map(lambda x: x if isinstance(x, list) else [])
            # формирование колонок с разницей между текущим временем и предыдущими строками
            for idx in range(1, self.time_shift_counts + 1):
                td_col = f'time_diff_{idx}'
                df[td_col] = df.progress_apply(
                    lambda row: self.make_shift_times(row, idx), axis=1)
                # Если делаем бинарные признаки
                if hasattr(self, 'time_shift_counts_binary'):
                    if self.time_shift_counts_binary:
                        df[td_col] = df[td_col].map(lambda x: x >= 0).astype(int)
            #
            self.comment['time_shift_counts'] = self.time_shift_counts
            #
            if hasattr(self, 'time_shift_counts_binary') and self.time_shift_counts_binary:
                self.comment['time_shift_counts_binary'] = self.time_shift_counts_binary
            # удалим временные колонки
            df.drop(['grp_times'], axis=1, inplace=True)

        # Добавляем колонку 'tag_id_total' как отношение кол-ва по tag_id к общему кол-ву
        grp_cols = ['user_id', 'tag_id']
        df['tag_id_total'] = df.groupby(grp_cols)['bid_ip'].transform('count').fillna(0)
        df['tag_id_total'] = df['tag_id_total'] / df['record_count']

        # Добавляем колонку 'tag_id_count' с количеством записей для каждого grp_cols,
        # минимальной датой, максимальной датой --> для расчета разницы во времени
        grp_cols = ['user_id', 'date', 'tag_id']
        df['tag_id_count'] = df.groupby(grp_cols)['bid_ip'].transform('count').fillna(0)
        df['tag_id_min'] = df.groupby(grp_cols)['time'].transform('min')
        df['tag_id_max'] = df.groupby(grp_cols)['time'].transform('max')
        df['tag_id_diff'] = (df['tag_id_max'] - df['tag_id_min']).dt.total_seconds().fillna(0)
        # удалим временные колонки
        df.drop(['date', 'tag_id_min', 'tag_id_max'], axis=1, inplace=True)

        self.exclude_columns.extend(['landing_page', 'battr', 'creative_size',
                                     'ud_cookie_ts', 'content_tags', 'unique_id',
                                     ])
        print_time(start_time)

        return df

    def fit(self, df, file_df=None):
        """
        Формирование фич
        :param df: исходный ФД
        :param file_df: Предобработанный Файл .pkl с полным путём
        :return: обработанный ДФ
        """

        self.comment['max_limit_nans'] = self.max_limit_nans

        if hasattr(self, 'concat_ssp_full_placement_id2'):
            if self.concat_ssp_full_placement_id2:
                self.comment['concat_ssp_full_placement_id2'] = True

        if hasattr(self, 'concat_user_id_bid_ip'):
            if self.concat_user_id_bid_ip:
                self.comment['concat_user_id_bid_ip'] = True

        df = self.initial_preparation(df)

        # удалять из трейна юзеров с пропусками
        if self.drop_nan_users:
            df = df[df.known > 0]
            self.comment['info'] = 'df[df.known > 0]'

        # Уберем пропущенных юзеров, которые встретились X раз
        if self.drop_nan_users_with_records:
            df = df[~((df.known == 0) &
                      (df.record_count <= self.drop_nan_users_with_records))]
            N = self.drop_nan_users_with_records
            self.comment['info'] = f'df[~((df.known == 0) & (df.record_count <= {N}))]'

        if self.vectorizer is not None:
            df = self.vectorizer_codes(df, fit_vectorizer=True)

        if hasattr(self, 'vectorizer_content_category'):
            if self.cont_cat_vectorizer is not None:
                df = self.vectorizer_content_category(df, fit_vectorizer=True)

        return df

    def vectorizer_codes(self, df, fit_vectorizer=False):
        """
        Векторизация турникетов за один день по group_columns
        :param df: исходный ДФ
        :param fit_vectorizer: Будем обучать self.vectorizer
        :return: обработанный ДФ
        """
        print(f'Векторизация последовательности user_segments')

        # максимальная длина последовательности mcc_codes
        max_gc = df.user_segments.str.len().max()
        if self.vector_limit:
            max_gc = min(self.vector_limit, max_gc)

        user_segments = df.user_segments.tolist()

        # Если векторайзер не был обучен - учим его
        if fit_vectorizer:
            self.bigram_vectorizer = self.vectorizer(ngram_range=self.ngram_range,
                                                     min_df=self.min_df,
                                                     max_features=self.max_features)
            self.bigram_vectorizer.fit(user_segments)

        bigram = self.bigram_vectorizer.transform(user_segments).toarray()

        del user_segments
        # Вызов сборщика мусора для освобождения памяти, занятой удаленными объектами
        gc.collect()

        print(f'Векторизация породила: {bigram.shape[1]} колонок')
        vct_columns = [f'vct_{n:03}' for n in range(bigram.shape[1])]
        print(f'df.shape {df.shape} bigram.shape {bigram.shape} nan {np.isnan(bigram).sum()}')
        df_bigram = pd.DataFrame(bigram, columns=vct_columns, index=df.index)
        df = pd.concat([df, df_bigram], axis=1)
        print(f'concat df.shape {df.shape} пропусков: {df.isna().sum().sum()}')

        del df_bigram
        # Вызов сборщика мусора для освобождения памяти, занятой удаленными объектами
        gc.collect()

        if fit_vectorizer:
            self.comment.update(vectorizer_user_segments=True,
                                vectorizer=self.vectorizer.__name__,
                                ngram_range=self.ngram_range,
                                min_df=self.min_df, max_features=self.max_features,
                                vector_limit=max_gc)
            self.numeric_columns.extend(vct_columns)
            self.exclude_columns.extend(['user_segments'])

        return df

    def vectorizer_content_category(self, df, fit_vectorizer=False):
        """
        Векторизация турникетов за один день по group_columns
        :param df: исходный ДФ
        :param fit_vectorizer: Будем обучать self.cont_cat_vectorizer
        :return: обработанный ДФ
        """
        print(f'Векторизация последовательности content_category')

        # максимальная длина последовательности mcc_codes
        max_gc = df.content_category.str.len().max()
        if self.cont_cat_vector_limit:
            max_gc = min(self.cont_cat_vector_limit, max_gc)

        content_category = df.content_category.tolist()

        # Если векторайзер не был обучен - учим его
        if fit_vectorizer:
            self.cont_cat_bigram_vectorizer = self.cont_cat_vectorizer(
                ngram_range=self.cont_cat_ngram_range,
                min_df=self.cont_cat_min_df,
                max_features=self.cont_cat_max_features)
            self.cont_cat_bigram_vectorizer.fit(content_category)

        bigram = self.cont_cat_bigram_vectorizer.transform(content_category).toarray()

        del content_category
        # Вызов сборщика мусора для освобождения памяти, занятой удаленными объектами
        gc.collect()

        print(f'Векторизация породила: {bigram.shape[1]} колонок')
        vct_columns = [f'vcc_{n:03}' for n in range(bigram.shape[1])]
        print(f'df.shape {df.shape} bigram.shape {bigram.shape} nan {np.isnan(bigram).sum()}')
        df_bigram = pd.DataFrame(bigram, columns=vct_columns, index=df.index)
        df = pd.concat([df, df_bigram], axis=1)
        print(f'concat df.shape {df.shape} пропусков: {df.isna().sum().sum()}')

        del df_bigram
        # Вызов сборщика мусора для освобождения памяти, занятой удаленными объектами
        gc.collect()

        if fit_vectorizer:
            self.comment.update(vectorizer_cont_cat=True,
                                cont_cat_vectorizer=self.cont_cat_vectorizer.__name__,
                                cont_cat_ngram_range=self.cont_cat_ngram_range,
                                cont_cat_min_df=self.cont_cat_min_df,
                                cont_cat_max_features=self.cont_cat_max_features,
                                cont_cat_vector_limit=max_gc)
            self.numeric_columns.extend(vct_columns)
            self.exclude_columns.extend(['content_category'])

        return df

    def transform(self, df, model_columns=None, after_fit=False):
        """
        Формирование остальных фич
        :param df: ДФ
        :param model_columns: список колонок, которые будут использованы в модели
        :param after_fit: Постобработка запущена после метода .fit()
        :return: ДФ с фичами
        """
        if not after_fit:
            df = self.initial_preparation(df)

            if self.vectorizer is not None:
                df = self.vectorizer_codes(df)

            if hasattr(self, 'cont_cat_vectorizer') and self.cont_cat_vectorizer is not None:
                df = self.vectorizer_content_category(df)

        total_time = print_msg('Постобработка данных...')

        if model_columns is None:
            model_columns = df.columns.to_list()

        exclude_columns = [col for col in set(self.exclude_columns) if col in df.columns]
        exclude_columns.extend(col for col in df.columns if col not in model_columns)

        if exclude_columns:
            df.drop(exclude_columns, axis=1, inplace=True)

        self.exclude_columns = sorted(exclude_columns)

        if isinstance(self.conv_info, pd.DataFrame):
            df = df.merge(self.conv_info, on='user_id', how='left').fillna(0)

        if hasattr(self, 'text_columns') and self.text_columns:
            text_columns = self.text_columns
        else:
            text_columns = []

        cat_features = list(df.columns[df.dtypes == 'object'])
        num_features = list(df.columns[~(df.dtypes == 'object')])

        cat_features = [f for f in cat_features if f not in ['label', 'time'] + text_columns]
        num_features = [f for f in num_features if f not in ['label', 'time'] + text_columns]

        # заполнение пропусков категориальных колонок
        df = self.fillna_fields(df, cat_features + text_columns)
        # заполнение пропусков цифровых колонок
        df = self.fillna_fields(df, num_features, numeric_fields=True)

        self.category_columns = cat_features
        self.numeric_columns = num_features

        df = self.cat_dummies(df)
        df = self.apply_scaler(df)

        print_time(total_time)

        # Переводим типы данных в минимально допустимые - экономим ресурсы
        df = memory_compression(df, exclude_columns=text_columns)

        return df

    def fit_transform(self, df, file_df=None, model_columns=None):
        """
        fit + transform
        :param df: исходный ФД
        :param file_df: Предобработанный Файл .pkl с полным путём
        :param model_columns: список колонок, которые будут использованы в модели
        :return: ДФ с фичами
        """
        df = self.fit(df, file_df=file_df)
        df = self.transform(df, model_columns=model_columns, after_fit=True)
        return df
