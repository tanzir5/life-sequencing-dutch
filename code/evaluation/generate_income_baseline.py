import pandas as pd
import pyreadstat
import pickle

# Get RINPERSONs of people who we want to check for income
df = pd.read_csv("RINPERSOON_and_income_30.txt",
                 delimiter="\t",
                 usecols=["RINPERSOON"],
                 dtype={'RINPERSOON': int, 'incomeAge30': int, 'birthYear': int})

user_set = set(df['RINPERSOON'])
print(len(user_set), flush=True)

baseline_by_years = {}
years = ['2011', '2012', '2013', '2014', '2015', '2016',
         '2017', '2018', '2019', '2020', '2021', '2022']
for year in years:
    df = pd.read_spss("INPA" + year + "TABV2.sav",
                      usecols=['RINPERSOON', 'INPBELI'])
    # dtype={"RINPERSOON": int, "INPPERSPRIM": int})

    # df = df.astype({"RINPERSOON": int, "INPPERSPRIM": int})
    user_list = list(df['RINPERSOON'])
    income_list = list(df['INPBELI'])

    yearly_baseline = {}

    for i in range(len(user_list)):
        try:
            user = int(user_list[i])
            income = income_list[i]
            if income == "9999999999":
                continue
            income = int(income)
        except:
            continue

        if user in user_set:
            yearly_baseline[user] = income

    baseline_by_years[year] = yearly_baseline

    with open("income_baseline_by_year.pkl", "wb") as pkl_file:
        pickle.dump(baseline_by_years, pkl_file)