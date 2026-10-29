import pickle

import numpy as np
import pandas as pd

from utils import const
from utils.utils import load_pickle


class Sampler(object):
    def __init__(self, data_path, search, user_vocab, session_items=None) -> None:
        self.search = search
        self.user_vocab = user_vocab
        self._session_items = session_items
        if isinstance(data_path, pd.DataFrame):
            self.data = data_path
        else:
            self.data = load_pickle(data_path)
        self._group_cache = {}

    @staticmethod
    def normalize_timestamp(timestamp):
        timestamp = float(timestamp)
        if np.isfinite(timestamp) and abs(timestamp) >= 1e11:
            timestamp /= 1000.0
        return int(timestamp) if np.isfinite(timestamp) else timestamp

    def _event_groups(self, user):
        """Reconstruct per-user action groups from paired public streams."""
        cached = self._group_cache.get(user)
        if cached is not None:
            return cached

        data = self.user_vocab[user]
        required = ('src_his', 'src_his_ts', 'src_session_his',
                    'src_session_his_ts')
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError('missing source-history fields for user {}: {}'.format(user, missing))
        source_lengths = [len(data[key]) for key in required]
        if len(set(source_lengths)) != 1:
            raise ValueError('misaligned source-history lengths for user {}: {}'.format(user, source_lengths))
        if any(self.normalize_timestamp(a) != self.normalize_timestamp(b) for a, b in zip(data['src_his_ts'], data['src_session_his_ts'])):
            raise ValueError('misaligned source-history timestamps for user {}'.format(user))
        if self._session_items is None:
            session_path = getattr(const, 'session_map_vocab', None)
            if not session_path:
                raise ValueError('source-session vocabulary is required for group reconstruction')
            self._session_items = load_pickle(session_path)['pos_items']
        for item, session_id in zip(data['src_his'], data['src_session_his']):
            try:
                session_items = np.asarray(self._session_items[int(session_id)])
            except (IndexError, KeyError, TypeError) as error:
                raise ValueError('unknown source session {} for user {}'.format(session_id, user)) from error
            if not np.any(session_items == int(item)):
                raise ValueError('source item {} is not in session {} for user {}'.format(item, session_id, user))
        rec_items = data['rec_his']
        rec_times = data['rec_his_ts']
        src_items = data['src_his']
        src_times = data['src_his_ts']
        rec_groups = [0] * len(rec_items)
        src_groups = [0] * len(src_items)
        group_by_key = {}
        next_group = 1
        for items, times in ((rec_items, rec_times), (src_items, src_times)):
            for item, timestamp in zip(items, times):
                key = (self.normalize_timestamp(timestamp), int(item))
                if key not in group_by_key:
                    group_by_key[key] = next_group
                    next_group += 1

        rec_groups = [group_by_key[(self.normalize_timestamp(timestamp), int(item))]
                      for item, timestamp in zip(rec_items, rec_times)]
        src_groups = [group_by_key[(self.normalize_timestamp(timestamp), int(item))]
                      for item, timestamp in zip(src_items, src_times)]
        result = (rec_groups, src_groups)
        self._group_cache[user] = result

        return result

    def _current_group_id(self, user, current_item, current_timestamp):
        if current_item is None:
            return 0
        rec_groups, src_groups = self._event_groups(user)
        data = self.user_vocab[user]
        key = (self.normalize_timestamp(current_timestamp), int(current_item))
        for items, times, groups in ((data['rec_his'], data['rec_his_ts'], rec_groups), (data['src_his'], data['src_his_ts'], src_groups)):
            for item, timestamp, group in zip(items, times, groups):
                if (self.normalize_timestamp(timestamp), int(item)) == key:
                    return group
        return 0

    def _current_group_from_session(self, user, session_id, current_timestamp):
        _, src_groups = self._event_groups(user)
        data = self.user_vocab[user]
        timestamp = self.normalize_timestamp(current_timestamp)
        for known_id, known_time, group in zip(data['src_session_his'], data['src_session_his_ts'], src_groups):
            if int(known_id) == int(session_id) and self.normalize_timestamp(known_time) == timestamp:
                return group
        return 0

    def sample(self, index):
        feed_dict = {}
        line = self.data.iloc[index]

        user = int(line['user_id'])
        feed_dict['user'] = [user]

        if 'candidate_items' in line.index:
            feed_dict['sample_id'] = [int(line['sample_id'])]
            feed_dict['candidate_items'] = [line['candidate_items']]
            current_item = next(
                (int(line[key]) for key in ('item_id', 'item', 'positive_item', 'pos_item', 'target_item')
                 if key in line.index and pd.notna(line[key])),
                None,
            )
        else:
            current_item = int(line['item_id'])
            feed_dict['item'] = [current_item]
            feed_dict['neg_items'] = [line['neg_items']]

        raw_timestamp = line['ts'] if 'ts' in line.index else line['timestamp']
        feed_dict['timestamp'] = [self.normalize_timestamp(raw_timestamp)]


        feed_dict['search'] = self.search
        if self.search:
            query = self.get_pad_query(line['keyword'])
            feed_dict['query'] = list([query])
            feed_dict['query_id'] = [int(line['search_session_id'])]

        rec_his_num = int(line['rec_his'])
        src_session_his_num = int(line['src_session_his'])
        feed_dict.update(
            self.get_all_his(user, rec_his_num, src_session_his_num))
        current_group_id = self._current_group_id(
            user, current_item, feed_dict['timestamp'][0])
        if current_group_id == 0 and self.search and 'search_session_id' in line.index:
            current_group_id = self._current_group_from_session(
                user, line['search_session_id'], feed_dict['timestamp'][0])
        feed_dict['current_group_id'] = [current_group_id]

        return feed_dict

    def get_all_his(self, user, rec_his_num, src_his_num):
        rec_his_item = self.user_vocab[user]['rec_his'][:rec_his_num][
            -const.max_rec_his_len:]
        rec_his_ts = self.user_vocab[user]['rec_his_ts'][:rec_his_num][
            -const.max_rec_his_len:]
        rec_groups, src_groups = self._event_groups(user)
        rec_his_group = rec_groups[:rec_his_num][-const.max_rec_his_len:]
        rec_his_ts = [self.normalize_timestamp(ts) for ts in rec_his_ts]
        rec_his_type = [1] * len(rec_his_item)
        rec_his = list(zip(rec_his_item, rec_his_ts, rec_his_type, rec_his_group))

        src_his_item = self.user_vocab[user]['src_session_his'][:src_his_num][
            -const.max_src_session_his_len:]
        src_his_ts = self.user_vocab[user]['src_session_his_ts'][:src_his_num][
            -const.max_src_session_his_len:]
        src_his_ts = [self.normalize_timestamp(ts) for ts in src_his_ts]
        src_his_group = src_groups[:src_his_num][-const.max_src_session_his_len:]
        src_his_type = [2] * len(src_his_item)
        src_his = list(zip(src_his_item, src_his_ts, src_his_type, src_his_group))

        sorted_all_his = sorted(
            rec_his + src_his, key=lambda x: x[1])[-const.max_all_his_len:]
        sorted_all_his += [(0, np.inf, 0, 0)] * (
            const.max_all_his_len - len(sorted_all_his))
        sorted_all_his_item = [x[0] for x in sorted_all_his]
        sorted_all_his_time = [x[1] for x in sorted_all_his]
        sorted_all_his_type = [x[2] for x in sorted_all_his]

        return {
            "all_his": [sorted_all_his_item],
            "all_his_ts": [sorted_all_his_time],
            "all_his_type": [sorted_all_his_type],
            "all_his_group_id": [[x[3] for x in sorted_all_his]]
        }

    def get_pad_query(self, query):
        if type(query) == str:
            query = eval(query)
        if type(query) == int:
            query = [query]
        query = query[:const.max_query_word_len]
        if len(query) < const.max_query_word_len:
            query += [0] * (const.max_query_word_len - len(query))
        return query
