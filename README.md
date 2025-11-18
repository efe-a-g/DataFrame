# DataFrame

The dataframe.py file implements an implementation of a DataFrame and Series data with increased 
type safety and more predictable behaviour with None's compared to other existing libraries such 
as pandas.

The underlying operations are written only using Python lists - this makes the code readable and reduces
dependencies. However, this means the code may be much slower in certain cases than other libraries such
as pandas which utilise numpy for efficient vectorised compiled code.
