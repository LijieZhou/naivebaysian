1. Observation:

For categorical attributes, see here: https://public.tableau.com/profile/lijie7852#!/vizhome/CensusIncomeorAdultdataset/NativeCountryLabel

For continuous attributes, see here: https://public.tableau.com/profile/lijie7852#!/vizhome/CensusIncomeorAdultdataset_2/Age_bin1

a. In terms of work class, self-emp-inc has a leading advantage on earning >50k: 55.7% of the self-emp-inc can earn more than 50k, compared to 38.6% for Federal-gov, 29% for local-gov and 21% for private. 
b. 74.1% of the population who earn a doctor degree can earn over 50k, compared to 55.7% of Master's and 41.5% of Bachelor's and 16.0% of high shcool grad. 
c. In terms of occupation, 48.4% of the exec-managerial and 44.9% of prof-specialty earn more than 50k. On the other hand, 0% priv-house-serv and 6.3% handlers-and-cleaners can earn over 50k. 
d. In terms of race, 26.6% of Asian and 25.6% of white earn over 50k compared to 12.4% Black. 
e. 10.9% of women earn over 50k compared to 30.6% of men. 
f. 61.4% of the people between age 50-59 earn over 50k, compared to 37.1% of age 40-49 and 26.8% of age 30-39. 
g. In terms of years of education, 73.7% of people who accept 12-15 years of education earn over 50k, compared to 42.3% for people who get education for 9-12 years and 17.9% of 6-9 years. 


2. Missing Data
There are three variables that have missing values. 
WorkClass: ? 1837 out of 32562 
Occupation: ? 1844 out of 32562
Native Country: ? 584 out of 32562

We ASSUME the data is missing completely at random (MCAR).

I follow this article to decide two different strategies to handle the missing value: 
https://liberalarts.utexas.edu/prc/_files/cs/Missing-Data.pdf

a. Listwise deletion

1) All 1837 records whose working class value are missing also have missing values in their occupation. Since this is relatively small compared to the whole dataset, I decided to remove these 1837 records. 
2) There are 7 records whose work class is “never worked” and “occupation” is “?”. I will remark their occupation “?” the same as their work class “never worked”. 
3) For the rest 30726 records, 557 records have missing value in native country, I will remove these records. 

The Listwise deletion ends with 30170 records.

b. Fill-in missing values with “Unknown”

1) All 1837 records whose working class value are missing are manually marked as “Unknown”. 
2) All 7 records whose work class is “never worked” and occupation is “?”, their occupation will be manually marked as “never worked”. 
3) All the records with native country “?” will be marked “unknown”

The “Unknown” approach ends with 32562 records. 

3. Implement Naive Bayes’ Classifier










