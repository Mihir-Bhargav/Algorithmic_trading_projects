# # # count = 0
# # # numbers = [10, 20, 30, 50, 25, 2, 34, 5, 99]
# # # for number in numbers:
# # #     increased = number * (1 + 0.03)
# # #     print(increased)
# # #     count +=1
# # #     print(count)


# # # #  xxxxxx
# # # import math
# # # print(list(range(0,10)))
# # # for i in range(13):
# # #     print(i**3 + i**2 + 34 * i)

# # import math
# # # Current_Value = 10000
# # # interest_rate = 3
# # # interest = Current_Value * interest_rate / 100
# # # FV = float(Current_Value * (1 + interest_rate / 100))
# # # print(FV)

# # # FV = float(5000 * (1.0435) ** 3)
# # # print(FV)

# # # FV = (CV * (1 + i) ** y)
# # # Rearragning this  formula gives: 
# # # CV = FV/ (1 + i) ** y

# # # FV = (5500 - 5000) / 4 
# # # CFV = (FV/ 5000) * 100
# # # print(f"The interest rate is {CFV} % ")

# # NPV = 0
# # cf = [-15000, 5000, 7000, 10000]
# # f = 1.05
# # for i in range(4):
# #    NPV += cf[i] / f**(i)
# #    print(NPV)

# # total = 0
# # for i in cf:
# #    total += i
# #    print(total)
# #    print(len(cf))

# # # # tuples are immutable
# # # from collections.abc import Mapping

# # # nums = (1, 2, 3, 1, 2, 1, 1, 1)
# # # enum = list(enumerate(nums))
# # # print(enum)
# # # tup = enum[0]
# # # print(tup)
# # # print(nums.count(1))
# # # print(nums.index(3))
# # # list1 = list(nums)
# # # print(list1)
# # # # You can convert list into a tuple and vice versa

# # # stock = ["NVDA", "AAPL", "MSFT"]
# # # price = [100, 200, 300]
# # # mapping = dict(zip(stock, price))
# # # print(type(mapping))
# # # print(mapping.keys())
# # # print(list(mapping.items()))
# # # dict(zip(stock, price))
# # # for i in zip(stock, price):
# # #   print(i)

# # # IO = 100
# # # IO = float(IO)
# # # print(type(IO))

# # # # interest_rate = input("Enter interest rate in %: ")
# # # # print(f"Interest rate is {interest_rate} % ")

# # # name = "mary"
# # # age = 20
# # # city = "New York"
# # # print(
# # #     f' Her name is {name}, she is {age} years old and her current residence is {city}'
# # # )

# # # cashflows = [1000, -500, 200, -300, 1500, -100, 0]

# # # positive_flows = []
# # # negative_flows = []

# # # for flow in cashflows:
# # #   if flow > 0:
# # #     positive_flows.append(flow)
# # #   elif flow < 0:
# # #     negative_flows.append(flow)

# # # print("Positive Cashflows:", positive_flows)
# # # print("Negative Cashflows:", negative_flows)

# # # # You can use the keyword pass for else statements.

# # # x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,]
# # # for i in x :
# # #   if(i > 10):
# # #     print(i)
# # #   else:
# # #     pass

# # # a = 10
# # # b = 5
# # # diff = 0
# # # while a > b:
# # #   print((a, b), "a is greater than b")
# # #   a -= 1
# # #   diff += 1
# # #   print(diff)

# # # number = 354
# # # add = 17
# # # target = 365835

# # # count = 0
# # # while number < target:
# # #     number += add
# # #     count += 1
# # # print(count)

# # import math
# # import random


# # def random_num():
# #   random_Num = random.random()
# #   random_num_real = int(math.floor((random_Num * 100) + 1))
# #   print(random_num_real)


# # random_num()


# # def shuffle_list():
# #   list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# #   random.shuffle(list)
# #   # Shuffle the list in place
# #   print(list)


# # shuffle_list()

# # # Difference between keyword arguments and positional arguments
# # # Positional arguments are passed to a function based on their order
# # # Keyword arguments are passed to a function with a keyword and a value


# # def describe_person(name, age, city):
# #   print(f"Name: {name}, Age: {age}, City: {city}")


# # # Positional arguments
# # describe_person("Alice", 30, "London")

# # # Keyword arguments
# # describe_person(name="Bob", age=25, city="Paris")

# # # In the first one, the order matters, while in the second one, the order does not matter.

# # # You can mix positional and keyword arguments, but positional arguments must come first
# # describe_person("Charlie", age=40, city="Tokyo")
# # # unpacking iterables
# # tup = (1, 2, 3, 4, 5)
# # a, b, c, d, e = tup
# # print(a)

# # # cf = [100, -500, 200, -300, 1500, -100, 0]
# # # r = 0.06
# # # def npv(r, cf):
# # #   npv = 0
# # #   for i, cashflow in enumerate(cf):
# # #     npv += cashflow / (1 + r)**i
# # #     print(npv)

# # # npv(r, cf)

# # cf = [100, -500, 200, -300, 1500, -100, 0]
# # r = 0.06


# # def npv(*args):
# #   r, cf = args
# #   npv = 0
# #   for i, cashflow in enumerate(cf):
# #     npv += cashflow / (1 + r)**i
# #     print(npv)


# # npv(r, cf)

# # # Scopes
# # NPV = 40
# # print(NPV)
# # # Local scope - inside a function stays inside a function
# # # If the same variable is defined outside the function, the function will use the local variable. This is called shadowing. Be careful with this while defining functions and debugging.
# # # You can specifically tell python to use the global variable by using the keyword global. This is not recommended as it can lead to bugs.

# # # def is_even():
# # #   try:
# # #       user_input = float(input("Enter a number: "))
# # #       print(user_input)
# # #       if user_input % 2 == 0:
# # #           print("Even")
# # #       else:
# # #           print("Odd")
# # #   except ValueError:
# # #       print("Invalid input. Please enter a numeric value.")
# # #       is_even()

# # PV = 100
# # f = 1.03
# # n = 2
# # FV = PV * math.pow(f, n)
# # print(FV)

# # from math import sqrt

# # print(sqrt(16))

# # import numpy as np

# # np.sqrt(FV // PV)

# # cf = [100, -500, 200, -300, 1500, -100, 0]
# # cf_a = np.array(cf)
# # print(cf_a - 20)
# # # Unlike normal python, this directly subratscts 20 from each element in the array.

# # my_list = [3, 4, 6, 8, 6]
# # import numpy as np

# # my_array = np.array(my_list)
# # print(my_array + 10)
# # print(my_array**2 + 27)

# # for i in my_array:
# #   if i > 5:
# #     print(i)

# # import numpy as np

# # my_array = np.array([4, 3, 6, 9, 0, 1, 5])
# # print(my_array[2:4])

# # cf = np.array([20, 50, 70, 100, 50])
# # n = np.array([1, 2, 3, 4, 5])
# # r = 0.06
# # f = 1 + r
# # final_value = (cf / f**n)
# # print(final_value)
# # NPV = np.sum(final_value)
# # if NPV > 0:
# #   print(NPV)
# #   print("Accept")

# # else:
# #   print(NPV)
# #   print("Reject")

# # import numpy as np

# # cf = np.array([300, 500, 1000, 2000])
# # cf.round(2)
# # n = np.array([1, 2, 3, 4])
# # r = 0.06
# # f = 1 + r
# # print(cf / f**n)
# # print(np.sum(cf / f**n).round(2))
# # # This was the simple way to calculate NPV also the correct one.

# # PV = (cf.sum()) - np.sum(cf / f**n)
# # # This calculates the present value of the cashflows.
# # print(PV)
# # # Good work!

# # cf = np.array([100, -500, 200, -300, 1500, -100, 0])
# # cf.round(2)
# # last_3 = cf[-3:]
# # last_3[0] = 55
# # print(cf)
# # # THis affect the original array. This is called a view.
# # # Using the copy method will create a new array.
# # last_3 = cf[-3:].copy()
# # print(last_3)
# # print(cf.max())
# # print(cf.min())
# # print(cf.argmax())
# # print(cf.cumsum())
# # # So useful!
# # # import matplotlib.pyplot as plt
# # # plt.figure(figsize=(10, 6))
# # # plt.plot(cf.cumsum())
# # # plt.show()

# # # import numpy as np
# # # cf = [100, -500, 200, -300, 1500, -100, 0]
# # # print(np.sqrt(cf))

# # # import numpy as np

# # # my_array = ([3.435, 6.45464, 7.56, 8.45643, 5.6765])

# # # my_array = np.array([3.435, 6.45464, 7.56, 8.45643, 5.6765])
# # # print(np.round(my_array, 2))
# # # print(my_array > 5)

# # # import numpy as np

# # # my_array = np.array([9, 12, 8, 22, 11, 7])
# # # greater_ten = my_array[my_array > 10]

# # # cf = np.array([100, -500, 200, -300, 1500, -100, 0])
# # # mask1 = cf > (-100)
# # # mask2 = np.abs(cf) > 90
# # # print(mask1)
# # # print(mask1 | mask2)
# # # print(cf[mask1 | mask2])
# # # print(~(mask1 | mask2))

# # # my_array = np.array([5, 17, 22, 27, 35, 46, 23, 5, 29])
# # # mask1 = my_array >= 20
# # # mask2 = my_array < 30

# # # new_array = my_array[mask1 & mask2]

# # # print(np.where(cf.cumsum() > 0))

# # # print(np.where(my_array > 28)[0][0])

# # # print(list(range(10)))
# # # print(np.arange(1, 11, 0.5))
# # # print(np.arange(0.01, 0.15, 0.001))
# # # print(np.linspace(0.01, 0.15, 100))
# # # print(np.linspace(0.01, 0.15, num = 15))
# # # x = np.linspace(0, 2 * np.pi, 100)
# # # print(x.size)
# # # y = np.sin(x)
# # # print(y)

# # # import matplotlib.pyplot as plt

# # # plt.figure(figsize=(8, 5))
# # # plt.plot(x, y)
# # # plt.title("Sine Wave" , fontsize = 20)
# # # plt.xlabel("x", fontsize = 15)
# # # plt.ylabel("y", fontsize = 15)
# # # plt.show()

# # # print(np.arange(1000))
# # my_array = (np.linspace(1, 1000, 101))
# # print(my_array)

# # project1 = np.array([100, -500, 200, -300, 1500, -100, 10])
# # project2 = np.array([100, -500, 200, 300, -1500, -100, 30])
# # project3 = np.array([100, 500, 200, 300, -1500, -100, -820])
# # all_projects = np.array([project1, project2, project3])
# # print(all_projects[0, 2])

# # print(np.array(project1))
# # print(all_projects[:, 0].sum())
# # # => This will print the first column of all projects. Even the sum. Useful!

# # # If all arrays are not of the same size, you can use the resize to make them the same size. This will add 0s to the end of the array. .resize(10, refcheck=False)

# # print(np.sum(all_projects, axis=0))
# # # axis = 1 => gives the sum of each row.
# # # Even more poweful! This will sum all the columns of all projects.
# # print(np.mean(all_projects, axis=0))
# # print(np.mean(all_projects))

# # # Pandas => Tabular data, stored in a dataframe.
# # import pandas as pd


# # import numpy as np
# # cp = np.array([1, 2, 3, 4 ,5])
# # cp1 = cp + 10
# # print(cp1)

# # print("Hello") 

# # import matplotlib.pyplot as plt
# # x = np.linspace(0, 2 * np.pi, 100)
# # print(x.size)
# # y = np.sin(x)
# # print(y)

# # plt.figure(figsize=(8, 5))
# # plt.plot(x, y)
# # plt.title("Sine Wave" , fontsize = 20)
# # plt.xlabel("x", fontsize = 15)
# # plt.ylabel("y", fontsize = 15)
# # plt.show() 

# import pandas as pd
# titanic = pd.read_csv("titanic.csv")
# # pd.options.display.min_rows = 5
# # print(titanic.info())
# # print(titanic.describe())
# # print(len(titanic))
# # print(titanic["age"].describe())
# # print(titanic["age"].sum())
# # print(titanic["age"].median())
# # print(titanic["age"].unique())
# # print(titanic["age"].nunique(dropna= False))
# # print(titanic["age"].value_counts(sort=True, dropna=True, ascending=False, normalize=False, bins=10).sum())
# # Bins is the amount of objects you want to have
# # copy = titanic["age"].copy()
# # copy[2] = 10
# # print(copy[2])
# # dic = {1:10, 3:25, 2:6, 3:36, 4:36, 5:2, 6:0, 7:None}
# # print(dic)
# # sales = pd.Series(dic)
# # # print(sales)
# # # print(sales.sort_values(ascending=True))
# # # print(sales.sort_values(ascending=False, na_position="last", inplace=False,))
# # # The exact same can be done for index using sort_index instead of sort_values. 
# # summer = pd.read_csv("summer.csv")
# # print(summer.columns)
# # summer = pd.read_csv("summer.csv", index_col= "Athlete")
# # # print(summer.index[0])
# # # print(summer.index[-1])
# # # print(summer.index[100:102])
# # # print(summer.index.tolist())
# # print(summer.reset_index())
# # # .get_loc finds the object with the argument
# # summer = pd.read_csv("summer.csv")
# # print(summer.set_index("Year"))
# # # summer.index.values[0] = 9999
# # # print(summer.index[0])


# # You can use .shape, ,size, .index , .info, .describe
# # However, you can't use the most of the normal python as these framworks only work if they ar numerical for functions like sum() or mean(). But you can use the:
# # Note: The argument 'numerical_only=True' requires pandas version 1.5.0 or newer.
# # print(titanic.mean(numeric_only = True))
# # # This workd
# # print(titanic.sum(numeric_only = True))
# # # print(titanic.mean(numeric_only = True)).sort_values().head(2) => for exampl
# # print(titanic.sort_values(by = "age", ascending=True))
# # print(titanic["age"])
# # print(titanic.age.equals(titanic["age"]))
# # df = pd.read_csv("titanic.csv")
# # fare = titanic["fare"]
# # print(fare )

# # summer = pd.read_csv("summer.csv")
# # # print(summer.info())
# # athlete = pd.read_csv("summer.csv", index_col= "Athlete")
# # # print(athlete)
# # # print(summer.iloc[0])
# # # print(summer.iloc[[1, 2, 3]]) =>
# # print(summer.iloc[1:4])
# # print(summer.iloc[1, 4]) 
# # # You can also slice the observation. Here, I have sliced only to show the athlete. 
# # print(summer.iloc[1, [0, 2, 5, 7]])
# # This printed more details, those asked for. 
# # this_dude = summer.loc["HERSCHMANN, Otto"]
# # print(this_dude)
# # print(summer)
# # print(summer.loc["PHELPS, Michael"])
# # df = pd.read_csv("titanic.csv")
# # last_rows = df.iloc[-10:, : 2]
# # print(last_rows)
# # new_index = ["Medal_No{}".format(i) for i in range(1, len(summer) + 1)]
# # print(new_index)
# # summer.index = pd.Index(new_index)
# # print(summer.head())
# # summer.rename(index ={" HAJOS, Alfred" : ' HAYOS, Alfred'}, inplace = True)
# # print(summer)


# import math as math
# import numpy as np
# # stock_trades = np.array([234.4 , 34.2, -132.5, 187.34, -142.21, 23.2, 123,21232, -128.9, 96.53, -12.3, 23.23, 42.21 ])
# # total_value = stock_trades.sum()
# # print(total_value.round(2))

# # count_positive_trades = 0
# # count_negative_trades = 0
# # for i in stock_trades:
# #     if i > 0:
# #         count_positive_trades += 1

# #     else:
# #         count_negative_trades += 1

# # denominator = count_negative_trades + count_positive_trades
# # if denominator == 0:
# #     win_ratio = None
# #     print(f" The win to loss ratio is {win_ratio} %")
# # else:
# #     win_ratio = round((count_positive_trades / denominator) * 100, 2)
# #     print(f" The win to loss ratio is {win_ratio} %")

# # summer.loc is not working for some reason. 
# # ...existing code...


# # --- Slicing and selecting with .loc and .iloc ---

# # .iloc uses integer positions (row, column)
# # print(summer.iloc[1:4])            # Rows 1 to 3, all columns
# # print(summer.iloc[1, 4])           # Row 1, column 4 (single value)
# # print(summer.iloc[1, [0, 2, 5, 7]])# Row 1, columns 0, 2, 5, 7

# # # .loc uses labels (row and column names)
# # # To use .loc with athlete names, set the index to "Athlete"
# # summer_by_athlete = pd.read_csv("summer.csv", index_col="Athlete")
# # print(summer_by_athlete.loc["PHELPS, Michael"])  # All rows for this athlete

# # # Slicing with .loc (label-based, inclusive)
# # print(summer_by_athlete.loc["PHELPS, Michael":"THOMAS, Petria"])  # From PHELPS to THOMAS (inclusive)

# # # Selecting specific columns with .loc
# # print(summer_by_athlete.loc["PHELPS, Michael", ["City", "Sport", "Event", "Medal"]])

# # # ...existing code...
 
# # df = pd.read_csv("summer.csv", index_col = "Athlete")
# # sliced = df.loc[["PAVIA, Automne", "OCHAL, Glenn"], ["Country", "Sport"]]

# # summer = pd.read_csv("summer.csv", index_col = "Athlete")
# # print(summer.loc["UNDA, Maider", "Event"])

# # print(summer[["Year" , "Medal"]])
# # print(summer.iloc[10:21])

# # # # titanic = csv_read(titanic.csv)
# # # print(titanic.sex.head(10))
# # # print(titanic.sex == "male")
# # # Prints true or false
# # print(titanic.loc[titanic.sex == "male"])
# # Use the .loc => fewer mistakes. 
# # mask1 = titanic.sex == "male"
# # print(mask1)
# # titanic_male = titanic.loc[mask1]
# # print(titanic.loc[mask1])
# # # Let's filter with several conditions
# # mask1 = titanic.sex == "male"
# # mask2 = titanic.age > 14
# # print(titanic[mask1 & mask2])
# # print(titanic[mask1 & mask2].head())
# # male_surv = titanic.loc[mask1 & mask2, ["survived", "pclass", "age", "sex"]]
# # mask1 = titanic.sex == "female"
# # mask2 = titanic.age < 14
# # female_surv = titanic.loc[mask1 | mask2, ["survived", "pclass", "sex", "age"]]
# # print(female_surv.describe())
# # print(mask1.head())
# # print(mask2.head())

# summer = pd.read_csv("summer.csv")
# og_1992 = summer.loc[summer["Year"] == 1992]
# # print(og_1992)
# # print(summer.Year.between(1960, 1969).head())
# # og_1960s = summer.loc[summer.Year.between(1960, 1969)]
# # print(og_1960s)
# # fav_games = summer.loc[summer["Year"].isin([1972, 1976, 1980])]
# # print(fav_games)
# # not_fav_games = summer.loc[~summer["Year"].isin([1972, 1976, 1980])]
# # print(not_fav_games)

# # NA values and missing values.
# import numpy as np 
# # sales = pd.read_csv("sales.csv", index_col = 0)
# # print(sales)
# # This has a missing value... 
# # But just giving a space removes the NaN and returns blank space.
# # sales = sales.fillna(" ")
# # print(sales)
# # Handling NA values
# # print(titanic.isna().sum())
# # print(titanic.notna().sum())
# # print(titanic.dropna().shape)
# # print(titanic.dropna(how = "all").shape)
# # print(titanic.dropna(axis = 1, how = "any").shape)
# # print(titanic.dropna(axis = 1, thresh = 500).shape)
# # print(titanic.loc[titanic.age.isna()])
# # mean_age = titanic.age.mean()
# # print(mean_age)
# # titanic.age.fillna(value = mean_age, inplace = True)
# # # This replaces all the NaN with the mean age. 
# # print(titanic.age)
# # # axis 0 stands for index and axis 1 stands for columns
# # # New code, UPDATED:
# # print(titanic.mean(axis = 0, numeric_only=True))
# # print(titanic.corr(numeric_only=True))
# # VERY USEFUL, ESPECIALLY WHEN YOU WILL START ML LATER!!!!!!!!!!!
# import pandas as pd
# titanic = pd.read_csv("titanic.csv")
# import matplotlib.pyplot as plt
# # print(titanic.plot(subplots= True, sharex = False, figsize = (15, 20)))
# # print(plt.show())

# # xticks = range(0, 901, 100)
# # yticks = range(0, 81, 10)
# # titanic.age.plot(figsize=(15,20), c="red", linestyle=":", xlim =(0, 900), ylim=(0, 80), xticks = xticks, yticks = yticks)
# # plt.title("Age Distribution of Titanic Passengers")
# # plt.xlabel("Passenger number")
# # plt.ylabel("Age")
# # plt.show()

# # plt.style.use("seaborn")
# # This is a histrogram. Idea: Use it for probability distrubutions by writing an algo ?
# # titanic.age.plot(kind="hist", figsize=(10, 6), bins=20, color="lightblue", edgecolor="black")
# # plt.show() 
# # titanic.age.hist(figsize=(10, 6))
# # plt.show() 
# # plt.figure(figsize=(10, 6))
# # plt.hist(titanic.age.dropna(), bins = 80, density=True, cumulative=True)
# # # The density parameter: If True, the result is a probability density, i.e. the area under the histogram integrates to 1 (if cumulative=False).
# # plt.show()
# # print(titanic.head())

# import pandas as pd
# titanic = pd.read_csv("titanic.csv")
# import matplotlib.pyplot as plt
# # titanic.plot(kind="scatter", x="age", y="fare", figsize=(10, 6), c="survived")
# # # You can make it 3 dimensional graph by adding the c=survived.
# # plt.show()
# # # To display the plot, use plt.show()
# # plt.show()
# import seaborn as sns
# # plt.figure(figsize=(10, 6))
# # sns.set(font_scale=2, palette = "viridis")
# # sns.countplot(data = titanic, x="sex", hue="pclass")
# # plt.show()
# # sns.stripplot(data=titanic, x ="sex", y='age', jitter=True, hue="pclass", dodge = True)
# # sns.violinplot(data=titanic, x ="sex", y='age', hue="pclass", dodge = True, split=False)
# # sns.barplot(data=titanic, x="pclass", y="age", hue="sex", dodge=True)
# # plt.show() 

# # Linear regression, VERY USEFUL AND IMPORTANT!!!!
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# # sns.set(font_scale=1)
# # sns.jointplot(data=titanic, x="age", y="fare", height=5, kind="reg", color=(2/255, 62/255, 138/255))
# # plt.show()
# # sns.lmplot(data=titanic, x="age", y="survived", aspect = 1, height = 6, col="sex", logistic = True)
# # sns.heatmap(pd.crosstab(titanic.sex, titanic.pclass), annot =False, fmt="d", cmap=None, vmax=None)
# # plt.figure(figsize=(8, 5))
# # sns.heatmap(titanic.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
# # plt.show()
# # # print(summer.drop(columns=["Sport", "Discipline"]))
# titanic_slice = titanic.iloc[:10, [2,3]]
# # print(titanic_slice)
# gbo = titanic_slice.groupby("sex")
# # print(gbo.groups)
# l = list(gbo)
# # print(l[0])
# # print(titanic_slice.loc[titanic_slice["sex"] == "male"])
# # for element in gbo:
# #     print(element[1])
# # print(summer.Country.nunique())
# # split1 = summer.groupby("Country")
# # l = list(split1)
# # print(len(l))
# # split2 = summer.groupby(by = ["Country", "Gender"])
# # l2 = list(split2)
# # print(len(l2))
# # print(l2[104][1])
# titanic_slice = titanic.iloc[:10, [2,3]]
# # print(titanic_slice.groupby("sex").mean())
# # print(titanic.groupby("sex").survived.sum())
# # print(titanic.groupby("sex")[["fare", "age"]].max())
# new_df = titanic.groupby("sex").mean(numeric_only = True)
# # print(new_df)
# # import matplotlib.pyplot as plt
# # plt.style.use("seaborn-v0_8")
# # new_df.plot(kind = "bar", subplots = True, figsize = (8,15), fontsize = 13)
# # plt.show()
# import matplotlib.pyplot as plt
# import seaborn as sns
# # bio = pd.read_csv("Bio_lab.csv")
# # print(bio)
# # print(bio.columns)

#      # Black background for the figure
# # sns.set_style("dark") 
# # sns.lmplot(
# #     data=bio,
# #     x="temperature",
# #     y="pH",
# #     aspect=1,
# #     height=9,
# #     scatter_kws={"color": (60/255, 179/255, 113/255)},
# #     line_kws={"color": (60/255, 179/255, 113/255)},
# # )

# # plt.title("Temperature vs. pH", fontsize=16)
# # plt.xlabel("Temperature (°C)", fontsize=12)
# # plt.ylabel("pH Level", fontsize=12)
# # plt.show()

# import math as math
# # Small projects as practice by the giving 
# # survival_rate = titanic.groupby("sex")["survived"].mean()
# # SR_m = math.floor(survival_rate["male"] * 100)
# # SR_f = math.floor(survival_rate["female"]*100)
# # print(f'The survival rate for men was {SR_m} %')
# # print(f'The survival rate for women was {SR_f} %') 
# # print(titanic["survived"])

# # print(titanic.groupby("sex")["age"].mean())
# # oldest_man = titanic[titanic["sex"] == "male"]["age"].max()
# # print(oldest_man)
# # youngest_women = titanic[titanic["sex"] == "female"]["age"].min()
# # difference_age = oldest_man - youngest_women
# # print(youngest_women)
# # print(difference_age)

# # Find a linear regression between survival rate and cost of ticket. 
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Calculate survival rate by deck
# # deck_SR = titanic.groupby("deck")["survived"].mean().reset_index()

# # # Plot
# # plt.figure(figsize=(8, 6))
# # sns.barplot(data=deck_SR, x="deck", y="survived", palette="viridis")

# # plt.title("Survival Rate by Deck", fontsize=16)
# # plt.xlabel("Deck", fontsize=12)
# # plt.ylabel("Survival Rate", fontsize=12)
# # plt.tight_layout()
# # plt.show()

# # # You must be careful when looking for data, they are for example space sensitiive.
# # # my_dict = {"Micheal, Phelps":22} 
# # # print("Micheal, Phelps") The missing space throws an error. 
# # # if you wanted to find the total medals won, you can't do this: 
# # my_dict["Micheal, Phelps"] + my_dict["Random, Guy"]
# # This throws the error can add strings, not int type. So in the database, make sure not to use "" for int values.

# # from ib_async import *  # type: ignore
# # util.startLoop()
# # ib = IB()
# # ib.connect()

# # contract = Forex("EURUSD")
# # contract = ib.qualifyContracts(contract)[0]  
# # data1 = ib.reqMktData(contract)

# # contract = Stock("AAPL", "SMART", "USD")
# # contract = ib.qualifyContracts(contract)[0]  
# # data2 = ib.reqMktData(contract)

# # contract = Stock("LHA", "SMART", "EUR")
# # contract = ib.qualifyContracts(contract)[0]  
# # data3 = ib.reqMktData(contract)

# # ib.sleep(2)  # Give time for data to populate
# # print(data1)

# # for i in range(10):
# #     print(round(data1.marketPrice(), 5), round(data2.marketPrice(), 3), round(data3.marketPrice(), 3))
# #     ib.sleep(1)


# # contract = Stock("AAPL", "SMART", "USD")
# # cds = ib.reqContractDetails(contract)
# # contracts = [cd.contract for cd in cds]
# # # print(cds[0].contracts)
# # print(util.df(contracts))

# import pandas as pd
# from ib_async import * # type: ignore
# util.startLoop()
# ib =IB()
# ib.connect()
# contract = Stock("AAPL", "SMART", "NYSE")
# print(contract)
# order = MarketOrder(action = "BUY", totalQuantity = 1)
# print(order)
# trade = ib.placeOrder(contract, order)
# print(trade)
# ib.sleep(5)
# print(trade.log)
# # The code is fine, but it is just that the US markets are closed and for some reason refuses to trade in the indian markets. 
# order = MarketOrder(action = "SELL", totalQuantity = 1)
# print(order)
# trade = ib.placeOrder(contract, order)
# ib.sleep(1)
# print(trade)
# # print(ib.trades()) => shows the trades you have made
# With this you can __wait until the order is either filled or canceled__

# order = MarketOrder("BUY", 1)
# trade = ib.placeOrder(contract, order)
# while not trade.isDone():
#     ib.waitOnUpdate()

# from ib_async import * 
# import pandas as pd
# import time

# util.startLoop()
# ib = IB()
# ib.connect()

# # --- 1. Define Forex Contract (EUR.USD is highly liquid) ---
# contract = Forex("EURUSD")  # This sets up EUR against USD on IDEALPRO
# ib.qualifyContracts(contract)

# # --- 2. Place Market Order (BUY 10k EUR) ---
# order = MarketOrder("BUY", 10000)  # 10,000 EUR = 0.1 lot
# trade = ib.placeOrder(contract, order)

# # --- 3. Wait Until Filled ---
# while not trade.isDone():
#     ib.waitOnUpdate()

# print("Order filled:")
# print(trade)

# # --- 4. Retrieve and Process Fills ---
# fills = ib.fills()
# print(f"\nNumber of fills: {len(fills)}")

# executions = [fs.execution for fs in fills if fs.execution]
# if executions:
#     fill_df = util.df(executions)[["execId", "time", "side", "cumQty", "avgPrice"]].set_index("execId")
#     fill_df["TradeValue"] = fill_df.side.apply(lambda x: 1 if x == "SLD" else -1) * fill_df.cumQty * fill_df.avgPrice
#     print("\n--- Fill Data ---")
#     print(fill_df)
#     print("\nTotal Trade Value:", fill_df.TradeValue.sum())
# else:
#     print("No executions found.")

# # --- 5. Retrieve PnL and Commissions ---
# commissions = [fs.commissionReport for fs in fills if fs.commissionReport]
# if commissions:
#     profit_df = util.df(commissions)[["execId", "currency", "commission", "realizedPNL"]].set_index("execId")
#     print("\n--- Profit Data ---")
#     print(profit_df)
#     print("Total Realized PnL:", profit_df.realizedPNL.sum())
# else:
#     print("No commission data found.")
    
# print(fill_df.groupby("side")["TradeValue"].sum())

from ib_async import *  # type: ignore
import pandas as pd
import numpy as np
import time as time 

ib = IB()
print(ib.connect())

start_time = time.time()

order = MarketOrder(action="BUY", totalQuantity=1)
contract = Forex("EURUSD")
trade = ib.placeOrder(contract, order)
print(trade)

sleep_time = 5
ib.sleep(sleep_time)

fills = ib.fills()
print(f"Number of fills: {len(fills)}")

executions = [fs.execution for fs in fills if fs.execution]
if executions:
    fill_df = util.df(executions)[["execId", "time", "side", "cumQty", "avgPrice"]].set_index("execId")
    fill_df["TradeValue"] = fill_df.side.apply(lambda x: 1 if x == "SLD" else -1) * fill_df.cumQty * fill_df.avgPrice
    print("\n--- Fill Data ---")
    print(fill_df)
    print("Total Trade Value:", fill_df.TradeValue.sum())
else:
    print("No executions found hence can't print trade value.")

# Define excecutions as ib.fills()
# if excecutions, convert it to a pandas dataframe, and set index as order number => makes sense, as it gives a nice label like 1, 2 ,3 and so on. 
# Use a lambda function for logic, use an analogy with positive x to represent money leaving the account and else statement to add money to the tradevalue . 
# Finally print the tradevalue

# ticker = ib.reqMktData(contract, snapshot=True)
# if ticker = 1.1(trade): 
#      order = MarketOrder("AAPL", "SMART", "NYSE")
#      contract =Stock(action="SELL", totalQuantity = 1)
#      ib.placeOrder(contract, order)
#      print(ib.fills())

positive_trades = 0
negative_trades = 0

for fill in fills:
    trade_value = fill.execution.shares * fill.execution.price
    print(f"Trade value: {trade_value}")

    if trade_value > 0:
        positive_trades += 1
    else:
        negative_trades += 1

denominator = positive_trades + negative_trades

if denominator == 0:
    win_ratio = 0
else:
    win_ratio = (positive_trades / denominator) * 100

print(f"{win_ratio:.2f}% is the call to sell ratio")

end_time = time.time()
elapsed_time = end_time - start_time - sleep_time
print(f'The elapsed time is: {round(elapsed_time, 2)} seconds')
fills = ib.fills()
commission_reports = [fs.commissionReport for fs in fills if fs.commissionReport]

if commission_reports:
    profit_df = util.df(commission_reports)[["execId", "currency", "commission", "realizedPNL"]].set_index("execId")
    print(f"{profit_df.realizedPNL.sum()} is the total profit in $.")
else:
    print("No transactions have been made, hence no profit.")

# Let's now look at historical data..... 
# from ib_async import *  # type: ignore
# import pandas as pd
# import numpy as np
# import time as time
# import seaborn as sns
# import matplotlib.pyplot as plt

# ib = IB() 
# print(ib.connect())

# contract = Forex("EURUSD")

# data = ib.reqHistoricalData(
#     contract,
#     endDateTime='',
#     durationStr='60 D',
#     barSizeSetting='1 day',
#     whatToShow='MIDPOINT',
#     useRTH=True
# )

# # Convert to DataFrame
# df = util.df(data)

# # Plot using seaborn
# plt.figure(figsize=(10, 5))
# sns.lineplot(data=df, x="date", y="close", label="Close Price", color="blue")
# plt.title("EURUSD - Historical Close Prices")
# plt.xlabel("Date")
# plt.ylabel("Price")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# contract = Stock("ORCL", "SMART", "USD")

# data = ib.reqHistoricalData(
#     contract,
#     endDateTime="",
#     durationStr='60 D',
#     barSizeSetting="1 day",
#     whatToShow='MIDPOINT',
#     useRTH=False
# )       

# df =util.df(data)
# # print(df)

# # Seaborn part 
# plt.figure(figsize=(10, 6))
# sns.lineplot(data=df, x="date", y="close", label="Close Price", color="red")
# plt.title("ORACLE INC - Historical Close Prices")
# plt.xlabel("Date")
# plt.ylabel("Price")
# plt.grid(True)
# plt.tight_layout()
# plt.show() 


# Getting to long, a new document name algo_trading2 has been created....

