from typing import TypeVar, Generic, Optional

T = TypeVar("T")

class Series(Generic[T]):
    # Initialise with input list of values, checking each value is of correct type or None
    def __init__(self, values: list, expected_type: type):
        self.base_type = expected_type
        if not all(isinstance(v, self.base_type) or v is None for v in values):
            raise TypeError(f"All elements must be of type {self.base_type.__name__} or None")
        self.values = values
        self.size = len(values)

    # Generate a Series of the correct type after an operation
    def create_series(self, values: list[Optional[T]], new_type: type) -> 'Series[T]':
        if new_type is bool:
            return SeriesBool(values) 
        elif new_type is int:
            return SeriesInt(values)  
        elif new_type is float:
            return SeriesFloat(values)  
        elif new_type is str:
            return SeriesStr(values)  
        else:
            raise TypeError("Unsupported base type for Series")
    
    # Override the square bracket operator
    # TODO: For SeriesBool indexing, use numpy arrays for efficiency
    def __getitem__(self, key: int|'SeriesBool') -> Optional[T]|'SeriesBool':
        if isinstance(key, int):
            return self.values[key]
        elif isinstance(key, SeriesBool):
            if key.size != self.size:
                raise ValueError("Series index does not match length of Series")
            # Construct a a new Series with all rows/values where the boolean Series is True (if k also rules out any items with None).
            new_rows = [v for v, k in zip(self.values, key.values) if k] 
            return self.create_series(new_rows, self.base_type) 
        else:
            raise TypeError("Key must be an integer or SeriesBool")
    
    # General method to override operations, propagating None values
    # TODO: If considering efficiency, we could achieve a significant speedup by internally using vectorised operations with numpy arrays (especially for all types here except strings).
    # For example, an operation such as np_array * 2 is much faster than [x*2 for x in py_list]
    # We would have to take care of None values carefully as for example in multiplication by two a None value would result in an error
    # If the array is not homogeneous (i.e., contains None values), using a masked array where we mask the None values could be a solution.
    def override_operation(self, other: 'Series[T]'|Optional[T], operation, custom_return_type: Optional[type]=None) -> 'Series':
        if custom_return_type is None:
            return_type= self.base_type
        else:
            return_type = custom_return_type
        if isinstance(other, Series):
            if self.size != other.size:
                raise ValueError("Cannot operate on Series of different lengths")
            if self.base_type != other.base_type:
                raise TypeError("Cannot operate on Series of different base types")
            new_rows = [operation(v, k) if v is not None and k is not None else None for v, k in zip(self.values, other.values)]
            return self.create_series(new_rows, return_type)
        elif isinstance(other, self.base_type) or other is None:
            new_rows = [operation(v, other) if v is not None else None for v in self.values]
            return self.create_series(new_rows, return_type)
        else:
            raise TypeError("Can only operate on Series with another Series or a value of same base type or None")
    
    # Override the equality operator
    # TODO: For efficiency, use numpy arrays as above, taking care of None values
    def __eq__(self, other: 'Series[T]'|Optional[T]) -> 'SeriesBool':
        return self.override_operation(other, lambda a, b: a==b, bool)
    
    # String representation for printing
    def __str__(self) -> str:
        return f"Series(type={self.base_type.__name__}, values={self.values})"
         
class SeriesBool(Series[bool]):
    # Using parent methods with updated typing
    def __init__(self, values: list[Optional[bool]]):
        super().__init__(values, bool)
    
    # TODO: For all of these operations, vectorised numpy operations could be used for efficiency
    def __and__(self, other: 'SeriesBool'|Optional[bool]) -> 'SeriesBool':
        return super().override_operation(other, lambda a, b: a and b)
    def __or__(self, other: 'SeriesBool'|Optional[bool]) -> 'SeriesBool':
         return super().override_operation(other, lambda a, b: a or b)
    def __xor__(self, other: 'SeriesBool'|Optional[bool]) -> 'SeriesBool':
         return super().override_operation(other, lambda a, b: a ^ b)
    def __invert__(self) -> 'SeriesBool':
         return SeriesBool([not v if v is not None else None for v in self.values])

class SeriesStr(Series[str]):
    # Using parent methods with updated typing
    def __init__(self, values: list[Optional[str]]):
        super().__init__(values, str)

class SeriesNum(Series[T]):
    # TODO: For all of the functions in this class, vectorised numpy operations could be used for efficiency
    def __gt__(self, other: 'SeriesNum[T]'|Optional[T]) -> 'SeriesBool':
        return self.override_operation(other, lambda a, b: a > b, bool)
    def __lt__(self, other: 'SeriesNum[T]'|Optional[T]) -> 'SeriesBool':
        return self.override_operation(other, lambda a, b: a < b, bool)
    def __ge__(self, other: 'SeriesNum[T]'|Optional[T]) -> 'SeriesBool':
        return self.override_operation(other, lambda a, b: a >= b, bool)
    def __le__(self, other: 'SeriesNum[T]'|Optional[T]) -> 'SeriesBool':
        return self.override_operation(other, lambda a, b: a <= b, bool)
    def __ne__(self, other: 'SeriesNum[T]'|Optional[T]) -> 'SeriesBool':
        return self.override_operation(other, lambda a, b: a != b, bool)
    
    def __add__(self, other: 'SeriesNum[T]'|Optional[T]) -> 'SeriesNum[T]':
        return self.override_operation(other, lambda a, b: a + b)
    def __sub__(self, other: 'SeriesNum[T]'|Optional[T]) -> 'SeriesNum[T]':
        return self.override_operation(other, lambda a, b: a - b)
    def __mul__(self, other: 'SeriesNum[T]'|Optional[T]) -> 'SeriesNum[T]':
        return self.override_operation(other, lambda a, b: a * b)
    def __truediv__(self, other: 'SeriesNum[T]'|Optional[T]) -> 'SeriesNum[T]':
        # Handle division by zero by returning None
        def safe_div(a, b):
            try:
                return a / b
            except Exception:
                return None
        return self.override_operation(other, safe_div, float)
    
    # Reverse operations to support cases where the Series is on the right side of the operator could be added here if needed.

class SeriesInt(SeriesNum[int]):
    # Using parent methods with updated typing
    def __init__(self, values: list[Optional[int]]):
        super().__init__(values, int)

class SeriesFloat(SeriesNum[float]):
    # Using parent methods with updated typing
    def __init__(self, values: list[Optional[float]]):
        super().__init__(values, float)

class DataFrame:
    # Initialise with a dictionary of Series, checking all Series are of the same length
    def __init__(self, data: dict[str, Series]):
        self.data = data
        if not all(isinstance(series, Series) for series in data.values()):
            raise TypeError(f"All values in DataFrame must be Series")
        sizes = [series.size for series in data.values()]
        if sizes!=[]:
            if not all(size == sizes[0] for size in sizes):
                raise ValueError("All Series in DataFrame must be of the same length")
            self.size = sizes[0]
        else:
            self.size = 0
    
    # Override the square bracket operator, accepting either a column name (str) or a boolean Series for filtering
    def __getitem__(self, key: str|SeriesBool) -> Series|'DataFrame':
        if isinstance(key, str):
            if key not in self.data:
                raise KeyError(f"Column '{key}' does not exist in DataFrame")
            return self.data[key]
        elif isinstance(key, SeriesBool):
            if key.size != self.size:
                raise ValueError("Series index does not match length of DataFrame")
            new_data = {col: series[key] for col, series in self.data.items()}
            return DataFrame(new_data)
        else:
            raise TypeError("Key must be a string or SeriesBool")

    # String representation for printing    
    def __str__(self) -> str:
        s = f"DataFrame(\n"
        for col, series in self.data.items():
            s += f"{col}: {series}\n"
        s += ")"
        return s

