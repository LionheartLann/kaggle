# learning Exploratory Data Analysis
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.INFO)

pd.options.display.max_rows = 1000
pd.options.display.max_columns = 20

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# first of all, have a look at the distribution of labels
#  train.SalePrice is equal to train['SalePrice']
sales = train.SalePrice
# plot hist 
bins = 100
plt.hist(sales, bins=bins)

# a glance at Size of living area 
plt.hist(train.GrLivArea, bins=bins)

# using train.head() to look at all features' value
logging.info('train head overview: %s', train.head())

#linear distance of street connected to property
#plt.hist(train.LotFrontage, bins=bins)

# Month sold
plt.hist(train.MoSold, bins=24)

# Missing data count & sort & plot
# the best visulization currently
missing = train.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar()

# look at the distribution of saleprice
import scipy.stats as st
import seaborn as sns
y = train['SalePrice']
plt.figure(1); plt.title('Johnson SU')
sns.distplot(y, kde=False, fit=st.johnsonsu)
plt.figure(2); plt.title('Normal')
sns.distplot(y, kde=False, fit=st.norm)
plt.figure(3); plt.title('Log Normal')
sns.distplot(y, kde=False, fit=st.lognorm)
