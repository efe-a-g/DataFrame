import unittest
import numpy as np
from dataframe import SeriesBool, SeriesStr, SeriesInt, SeriesFloat, DataFrame

class Test(unittest.TestCase):
    def test_df_type_mismatch(self):    
        with self.assertRaises(TypeError):
            df = DataFrame({"ID": [2]})
    
    def test_type_mismatch(self):
        with self.assertRaises(TypeError):
            price = SeriesInt([77, "89", 66, 55])
    
    def test_length_mismatch_series(self):
        with self.assertRaises(ValueError):
            price = SeriesFloat([7.0,3.5,8.0])
            sales = SeriesInt([5,3,1,10])
            result = price + sales
    
    def test_length_mismatch_df(self):
        idx = SeriesStr(["A", "B", "D","X"])
        price = SeriesFloat([7.0,3.5,8.0])

        with self.assertRaises(ValueError):
            df = DataFrame({"ID": idx,"price": price})
    
    def test_division_by_zero(self):
        sales = SeriesInt([10, 0, 5, None])
        result = sales / 0
        self.assertEqual(result.values, [None, None, None, None])
    
    def test_none_propagation(self):
        sales = SeriesInt([10, None, 5, 2])
        result = sales + 5
        self.assertEqual(result.values, [15, None, 10, 7])
    
    def test_boolean_indexing_length_mismatch(self):
        sales = SeriesInt([10, 5, 2])
        bool_index = SeriesBool([True, False])
        with self.assertRaises(ValueError):
            result = sales[bool_index]
    
    def test_unsupported_type(self):
        with self.assertRaises(TypeError):
            def custom_operation(a, b):
                return [a + b]
            sales = SeriesInt([10, 5, 2])
            result = sales.override_operation(5, custom_operation, list)    
    
    def test_invalid_key_type(self):
        sales = SeriesInt([10, 5, 2])
        with self.assertRaises(TypeError):
            result = sales["invalid_key"]
    
    def test_different_type_item_operation(self):
        sales = SeriesInt([10, 5, 2])
        with self.assertRaises(TypeError):
            result = sales + "invalid_type"
    
    def test_different_type_series_operation(self):
        sales = SeriesInt([10, 5, 2])
        price = SeriesFloat([7.0,3.5,8.0])
        with self.assertRaises(TypeError):
            result = sales + price
    
    def test_repr(self):
        sales = SeriesInt([10, 5, None, 2])
        self.assertEqual(str(sales), "Series(type=int, values=[10, 5, None, 2])")
    
    def test_square_bracket(self):
        sales = SeriesInt([10, 5, None, 2])
        self.assertEqual(sales[0], 10)
        self.assertEqual(sales[2], None)
        bool_index = SeriesBool([True, False, True, False])
        result = sales[bool_index]
        self.assertEqual(result.values, [10, None])
    
    def test_bool_op(self):
        sales = SeriesInt([10, 5, None, 2])
        result = (sales > 3) & (sales < 10)
        self.assertEqual(result.values, [False, True, None, False])
        self.assertEqual(sales[result].values, [5])
    
    def test_str_equality(self):
        s1 = SeriesStr(["X4", "T3", "F8","C7"])
        s2 = SeriesStr(["X4", "T3", "F8","C8"])
        self.assertEqual((s1 == s2).values, [True, True, True, False])
    
    def test_str_access(self):
        s = SeriesStr(["A1", "B2", "C3","D4"])
        self.assertEqual(s[1], "B2")
        bool_index = SeriesBool([False, True, False, True])
        result = s[bool_index]
        self.assertEqual(result.values, ["B2", "D4"])
    
    def test_str_type_mismatch(self):
        with self.assertRaises(TypeError):
            s = SeriesStr(["A", 123, "C","D"])
    
    def test_arithmetic(self):
        s = SeriesInt([10, 5, None, 2])
        self.assertEqual((s + 5).values, [15, 10, None, 7])
        self.assertEqual((s - 2).values, [8, 3, None, 0])
        self.assertEqual((s * 3).values, [30, 15, None, 6])
        self.assertEqual((s / 2).values, [5.0, 2.5, None, 1.0])
        self.assertEqual((s == 5).values, [False, True, None, False])
        self.assertEqual((s != 5).values, [True, False, None, True])
        self.assertEqual((s < 5).values, [False, False, None, True])
        self.assertEqual((s <= 5).values, [False, True, None, True])
        self.assertEqual((s > 5).values, [True, False, None, False])
        self.assertEqual((s >= 5).values, [True, True, None, False])
    
    def test_df_repr(self):
        df = DataFrame({
            "ID": SeriesStr(["A1", "B2"]),
            "price": SeriesFloat([7.0,3.5]),
            "sales": SeriesInt([5,3]),
            "taxed": SeriesBool([False, False])
        })
        expected = ("DataFrame(\n"
                    "ID: Series(type=str, values=['A1', 'B2'])\n"
                    "price: Series(type=float, values=[7.0, 3.5])\n"
                    "sales: Series(type=int, values=[5, 3])\n"
                    "taxed: Series(type=bool, values=[False, False])\n)")
        self.assertEqual(str(df), expected)
    
    def test_df_one(self):
        df = DataFrame({
            "Course": SeriesStr(["Maths", "Physics","CS","Engineering","Biology","History"]),
            "Lectures": SeriesInt([16,8,8,16,8,None]),
            "Exam": SeriesBool([True, True, False, False, False, None]),
            "Credits": SeriesInt([5,5,5,5,10, None])
        })
        result = df[(df["Lectures"] < 16) & (df["Credits"] >=5)]["Course"]
        self.assertEqual(result.values, ["Physics", "CS", "Biology"])
    
    def test_df_two(self):
        cube_minus_one = [i**3-1 for i in range(1,11)]
        primes = [n for n in range(2, 100) if all(n % i != 0 for i in range(2, int(n**0.5)+1))]

        df = DataFrame({
            "Numbers": SeriesInt(list(range(1,101))),
            "CubeMinusOne": SeriesBool([num in cube_minus_one for num in range(1,101)]),
            "Primes": SeriesBool([num in primes for num in range(1,101)])
        })

        result = df[df["CubeMinusOne"] & df["Primes"]]["Numbers"]
        self.assertEqual(result.values, [7])

    def test_df_three(self):
        serial = SeriesStr(np.random.randint(100000,999999,size=100).astype(str).tolist())
        price = SeriesFloat(np.random.randn(100).tolist())
        sales = SeriesInt(np.random.randint(1,20,size=100).tolist())

        df = DataFrame({"ID": serial,"price": price,"sales": sales})
        result1 = df[~((df["price"] <= 7.0) & (df["sales"] > 1))]["ID"]
        result2 = df[~(df["price"] <= 7.0) | ~(df["sales"] > 1)]["ID"]
        self.assertTrue(result1.values==result2.values)

if __name__ == "__main__":
    unittest.main()
    