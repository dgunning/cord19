import re
import pandas as pd
from functools import partial
import pendulum

YYYY_MON_DD = '\d{4} \w{3} \d{1,2}$'
YYYY_MON = '\d{4} \w{3}$'
YYYY = '\d{4}$'
YYYY_MM_DD = '\d{4}\-\d{2}\-\d{2}$'


def format_date(date, format):
    try:
        return pd.to_datetime(date, format=format)
    except ValueError:
        if re.match(YYYY_MON_DD, date):
            date = f'{date[:8]} 28'
            return pd.to_datetime(date, format=format)
        return f'ValueError'


def repair_date(date_column: pd.Series):
    mdates = date_column.copy().to_frame().fillna('')
    print("Fixing dates that are a list e.g. \"['2020-02-05', '2020-02']\"")
    idx_list = mdates.publish_time.str.match("\[.*")
    mdates.loc[idx_list, 'publish_time'] = mdates.loc[idx_list] \
        .publish_time.apply(lambda d: d[2:12])

    print('Fixing dates with the seasons e.g. "2014 Autumn"')
    idx_seasons = mdates.publish_time.str.match('.*(Spring|Summer|Fall|Autumn|Winter)')
    mdates.loc[idx_seasons, 'publish_time'] = mdates.loc[idx_seasons].publish_time \
        .str.replace('Spring', 'Apr 01') \
        .str.replace('Summer', 'Jul 01') \
        .str.replace('Autumn', 'Oct 01') \
        .str.replace('Fall', 'Oct 01') \
        .str.replace('Winter', 'Dec 21')

    print('Fixing dates like "2016 Nov 9 Jan-Feb"')
    idx_YYYY_MON_DD_extra = mdates.publish_time.str.match('\d{4} \w{3} \d{1,2}.+$')
    mdates.loc[idx_YYYY_MON_DD_extra, 'publish_time'] = \
        mdates.loc[idx_YYYY_MON_DD_extra].publish_time.apply(lambda d: d[:11].strip())

    print('Fixing dates like "2012 Jan-Mar"')
    idx_YYYY_MON_MON = mdates.publish_time.str.match('\d{4} \w{3}-\w{3}$')
    mdates.loc[idx_YYYY_MON_MON, 'publish_time'] = \
        mdates.loc[idx_YYYY_MON_MON].publish_time.apply(lambda d: d[:8].strip())

    print('Converting dates like "2020 Apr 13"')
    idx_YYYY_MON_DD = mdates.publish_time.str.match(YYYY_MON_DD, case=False)
    mdates.loc[idx_YYYY_MON_DD, 'publish_date'] = \
        mdates.loc[idx_YYYY_MON_DD, 'publish_time'].apply(partial(format_date, format='%Y %b %d'))

    print('Converting dates like "2020 Apr"')
    idx_YYYY_MON = mdates.publish_time.str.match(YYYY_MON, case=False)
    mdates.loc[idx_YYYY_MON, 'publish_date'] = \
        mdates.loc[idx_YYYY_MON, 'publish_time'].apply(partial(format_date, format='%Y %b'))

    print('Converting dates like "2020"')
    idx_YYYY = mdates.publish_time.str.match(YYYY, case=False)
    mdates.loc[idx_YYYY, 'publish_date'] = \
        mdates.loc[idx_YYYY, 'publish_time'].apply(partial(format_date, format='%Y'))

    print('Converting dates like "2020-01-21"')
    idx_YYYY_MM_DD = mdates.publish_time.str.match(YYYY_MM_DD, case=False)
    mdates.loc[idx_YYYY_MM_DD, 'publish_date'] = \
        mdates.loc[idx_YYYY_MM_DD, 'publish_time'].apply(partial(format_date, format='%Y-%m-%d'))

    return mdates.publish_date


def date_diff(date):
    if pd.isnull(date):
        return ''
    timestamp = date.timestamp()
    if timestamp < 0:
        diff = f'more than {pendulum.from_timestamp(0).diff_for_humans()}'
    else:
        diff = pendulum.from_timestamp(timestamp).diff_for_humans()
    return diff


def fix_dates(data, old_date_column='publish_time', new_date_column='published'):
    data[new_date_column] = repair_date(data[old_date_column])
    return data.drop(columns=[old_date_column])


def add_date_diff(data, date_column='published', new_column='when'):
    data[new_column] = data[date_column].apply(date_diff)
    return data
