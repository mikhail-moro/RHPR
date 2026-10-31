import importlib.util
from pathlib import Path
import unittest

import numpy as np


path = Path(__file__).resolve().parents[1] / 'scripts/prepare_movielens_text_ladder.py'
spec = importlib.util.spec_from_file_location('ml_text_preparation', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class MovieTextPreparationTests(unittest.TestCase):
    def test_person_identity_and_ranking_not_name_deduplication(self):
        people = [{'id': 3, 'name': 'A'}, {'id': 3, 'name': 'A alternate'},
                  {'id': 4, 'name': 'A'}, {'id': 5, 'name': 'B'}]
        self.assertEqual(module.named(people), ['A', 'A', 'B'])

    def test_timestamp_tie_is_distinct_from_duplicate_rating(self):
        # User1 has one first group and two heldout groups. User2 has only two
        # groups and must not contribute invented temporal train observations.
        ratings = np.array([[1, 10, 5, 1], [1, 11, 3, 1], [1, 12, 4, 2],
                            [1, 13, 4, 3], [1, 14, 2, 3],
                            [2, 15, 4, 1], [2, 16, 4, 2]])
        report = module.audit_timestamps(ratings)
        self.assertEqual(report['duplicate_user_movie_pairs'], 0)
        self.assertEqual(report['users_ordinary_last_two_share_timestamp'], 1)
        self.assertEqual(report['users_fewer_than_three_timestamps'], 1)
        self.assertEqual(report['group_embargo_train_events_eligible_users'], 2)
        self.assertEqual(report['strict_past_only_train_targets_eligible_users'], 0)
        self.assertEqual(report['extra_heldout_group_ratings_for_users_with_three_groups'], 1)

    def test_ladder_does_not_put_keywords_or_people_in_earlier_stages(self):
        fields = dict(title='A movie', genres=['Drama'], year=1995,
                      runtime='110 minutes', keywords=['mystery'], cast=['Actor A'],
                      directors=['Director B'])
        texts = module.field_texts(fields)
        self.assertNotIn('1995', texts[0])
        self.assertNotIn('mystery', texts[3])
        self.assertNotIn('Actor A', texts[4])
        self.assertNotIn('Director B', texts[5])
        self.assertIn('Director B', texts[6])


if __name__ == '__main__':
    unittest.main()
