import numpy as np
import pandas as pd
import statsmodels.stats.api as sms
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.proportion import proportions_ztest, proportion_confint

effect_size = sms.proportion_effectsize(0.13, 0.15)

# Расчет необходимого размера выборки
required_n = sms.NormalIndPower().solve_power(
    effect_size,
    power=0.8,
    alpha=0.05,
    ratio=1
)
print(f"required_n = {required_n}")
# чтение данных из csv файла
df = pd.read_csv('ab_data.csv')

# выведем первые 10 строк
print("---------------------------------Head--------------------------------------")
print(df.head())
print("----------------------------------------------------------------------------")
print("---------------------------------Info--------------------------------------")
print(df.info())
print("----------------------------------------------------------------------------")
print("--------------------Проверяем наличие подглядываний-------------------------")
# Проверим, не было ли "подглядываний"
print(pd.crosstab(df.group, df.landing_page))
print("----------------------------------------------------------------------------")
print(" ")
# print(t_end_1 - t_start)

print("--------------------Убеждаемся, что подглядываний нет----------------------")
df1 = df[((df['group'] == 'control') & (df['landing_page'] == 'old_page'))]
df2 = df[((df['group'] == 'treatment') & (df['landing_page'] == 'new_page'))]
df = pd.concat([df1, df2], ignore_index=True)

print(pd.crosstab(df.group, df.landing_page))
print("----------------------------------------------------------------------------")

print("--------Проверяем наличие пользователей, встречающихся более раза-----------")
print(df['user_id'].value_counts().sort_values())
print("----------------------------------------------------------------------------")
df = df[~df.duplicated('user_id')]
print("--------Убеждаемся, что нет пользователей, встречающихся более раза---------")
print(df['user_id'].value_counts().sort_values())
print("----------------------------------------------------------------------------")
# Теперь данные чистые
# Сформируем контрольную группу
control_sample = df[df['group'] == 'control'].sample(n=int(required_n), random_state=22)

# Сформируйте целевую группу такого же размера и с тем же random_state
treatment_sample = df[df['group'] == 'treatment'].sample(n=int(required_n), random_state=22)  # type your code here

# Объединим полученные наборы данных
ab_test = pd.concat([control_sample, treatment_sample], axis=0)
ab_test.reset_index(drop=True, inplace=True)

# Проверьте, что в ab_test все в порядке. Нет пустых записей, дублей, подглядываний, размеры соответствуют required_n

print("------------------------Проверка на пустые записи:--------------------------")
print(ab_test[ab_test.isna().any(axis='columns')])
print("--------------------------Проверка на дубли:--------------------------------")
print(ab_test[ab_test.duplicated('user_id')])
print("-----------------------Проверка на подглядывания:---------------------------")
print(pd.crosstab(ab_test.group, ab_test.landing_page))
print("------------------Проверка на соответствие required_n-----------------------")
print(int(required_n) * 2 == ab_test.shape[0])

print("----------------------Визуализация результатов------------------------------")
conversion_rates = ab_test.groupby('group')['converted'].describe()
conversion_rates['std_err'] = (conversion_rates['std'] / np.sqrt(conversion_rates['count']))
print(conversion_rates)
plt.figure(figsize=(8, 6))

sns.barplot(x=ab_test['group'], y=ab_test['converted'], errorbar=('ci', False))

plt.ylim(0, 0.17)
plt.title('Conversion rate by group', pad=20)
plt.xlabel('Group', labelpad=15)
plt.ylabel('Converted (proportion)', labelpad=15);
plt.show()
print("----------------------------------------------------------------------------")
print("---------------------------Проверка гипотезы--------------------------------")
control_results = ab_test[ab_test['group'] == 'control']['converted']
treatment_results = ab_test[ab_test['group'] == 'treatment']['converted']

# Рассчитаем значения z статистики и p значение с помощью библиотеки statsmodels:
successes = [control_results.sum(), treatment_results.sum()]
n_treat = treatment_results.count()
n_con = n_treat
nobs = [n_con, n_treat]

z_stat, p_val = proportions_ztest(successes, nobs=nobs)
(lower_con, lower_treat), (upper_con, upper_treat) = proportion_confint(successes, nobs=nobs, alpha=0.05)

print(f'z statistic: {z_stat:.2f}')
print(f'p-value: {p_val:.3f}')
print(f'confidence interval 95% for control group: [{lower_con:.3f}, {upper_con:.3f}]')
print(f'confidence interval 95% for treatment group: [{lower_treat:.3f}, {upper_treat:.3f}]')
print("\n")
print("------------------------------Вывод-----------------------------------------")
print(
    "Как видим, значение p-value > 0.05, следовательно, нулевая гипотеза не может\n"
    "быть отвергнута, а значит нововведение не увеличит конверсию.")
