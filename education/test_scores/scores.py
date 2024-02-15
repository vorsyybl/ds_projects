import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

df = pd.read_csv('dfs/StudentsPerformance.csv')
perf = df.loc[:, ['math score', 'reading score', 'writing score', 'race/ethnicity', 'parental level of education']]
perf = perf.rename(columns={
    'math score': 'math_score',
    'reading score': 'reading_score',
    'writing score': 'writing_score',
    'parental level of education': 'parental_edu'
})
#   CHECKING LINEARITY ASSUMPTION BETWEEN CVs AND DV
fig, axes = plt.subplots(3, 2, figsize=(15, 10))
axes = axes.flatten()
sns.regplot(ax=axes[0], x='reading_score', y='math_score', data=perf)
axes[0].set_title('Reading (CV) vs Math Score (DV)')
sns.regplot(ax=axes[1], x='writing_score', y='math_score', data=perf)
axes[1].set_title('Writing (CV) vs Math Score (DV)')
sns.regplot(ax=axes[2], x='math_score', y='reading_score', data=perf)
axes[2].set_title('Math (CV) vs Reading Score (DV)')
sns.regplot(ax=axes[3], x='writing_score', y='reading_score', data=perf)
axes[3].set_title('Writing (CV) vs Reading Score (DV)')
sns.regplot(ax=axes[4], x='math_score', y='writing_score', data=perf)
axes[4].set_title('Math (CV) vs Writing Score (DV)')
sns.regplot(ax=axes[5], x='reading_score', y='writing_score', data=perf)
axes[5].set_title('Reading (CV) vs Writing Score (DV)')
plt.show()

print(perf.info())
#   HOMOEGENEITY OF VARIANCES BETWEEN CATEGORIES
stat, p = stats.levene(*[group['math_score'].values for name, group in perf.groupby('parental_edu')])
print('Levene’s Test Statistic:', stat)
print('P-Value:', p)

#   NORMAL DISTRIBUTION OF RESIDUALS CHECK
math_model = ols('math_score ~ reading_score + writing_score + C(parental_edu)', data=perf).fit()
math_residuals = math_model.resid
sm.qqplot(math_residuals, line='45', fit=True)
plt.show()
print(math_model.summary())

reading_model = ols('reading_score ~ C(parental_edu) + math_score + writing_score', data=perf).fit()
reading_residuals = reading_model.resid
sm.qqplot(reading_residuals, line='45', fit=True)
plt.show()
print(reading_model.summary())

writing_model = ols('writing_score ~ C(parental_edu) + math_score + reading_score', data=perf).fit()
writing_residuals = writing_model.resid
sm.qqplot(writing_residuals, line='45', fit=True)
plt.show()
print(writing_model.summary())



