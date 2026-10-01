"""Exact short-sequence descriptors and conditional complete-product accounting."""
import math
import unittest

from scripts.capability.generation.describe_generation_products import describe,sequence_descriptors


class ProductDescriptorTests(unittest.TestCase):
    def test_composition_and_repetition(self):
        homogeneous=sequence_descriptors('AAAAA')
        self.assertEqual(homogeneous['composition_entropy'],0.)
        self.assertEqual(homogeneous['longest_identical_run_fraction'],1.)
        self.assertAlmostEqual(homogeneous['unique_3mer_fraction'],1/3)
        diverse=sequence_descriptors('ACDE')
        self.assertAlmostEqual(diverse['composition_entropy'],math.log(4))
        self.assertEqual(diverse['longest_identical_run_fraction'],.25)
        self.assertEqual(diverse['unique_3mer_fraction'],1.)
        self.assertIsNone(sequence_descriptors('AC')['unique_3mer_fraction'])

    def test_complete_strata_and_undefined_short_products(self):
        rows=[dict(sequence=s,residue_length=len(s),native_complete=complete,any_profile_hit=hit)
              for s,complete,hit in [('AC',True,False),('ACDE',True,True),('AAAA',False,False)]]
        result=describe(rows)
        self.assertEqual((result['n_attempts'],result['n_complete'],result['n_excluded_incomplete']),(3,2,1))
        complete=result['groups']['all_complete']['metrics']
        self.assertEqual(complete['residue_length']['quantiles']['median'],3.)
        self.assertEqual(complete['unique_3mer_fraction']['n_undefined'],1)
        nohit=result['groups']['complete_without_pfam']['metrics']['unique_3mer_fraction']
        self.assertEqual(nohit['n_defined'],0)
        self.assertIsNone(nohit['quantiles'])
        rows[0]['residue_length']=3
        with self.assertRaisesRegex(ValueError,'length mismatch'):describe(rows)
        rows[0]['native_complete']=None
        with self.assertRaisesRegex(ValueError,'missing'):describe(rows)


if __name__=='__main__':unittest.main()
