#!/usr/bin/env python
import unittest


# test case
# assertEqual, assertNotEqual, assertTrue, assertFalse
# assertIs, assertIsNot, assertIsNone, assertIsNotNone
# assertIn, assertNotIn, assertIsInstance, assertIsNotInstance
# assertRaises, assertRaisesRegexp
class TestMethods(unittest.TestCase):
    def setUp(self):
        self.text = "hello world"

    def tearDown(self):
        self.text = None

    # skip test: skip, skipIf, skipUnless
    @unittest.skip("skip this test")
    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_split(self):
        self.assertEqual(self.text.split(), ['hello', 'world'])
        with self.assertRaises(TypeError):
            self.text.split(2)

    @unittest.expectedFailure
    def test_fail(self):
        self.assertTrue(1 == 2, "test failed expectedly")

if __name__ == '__main__':
    # unittest.main()
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMethods)
    unittest.TextTestRunner(verbosity=2).run(suite)
