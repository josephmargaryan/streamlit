import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.metrics import accuracy_score


# Define function for page 1
def page1():
    # Read in the data
    df = pd.read_csv('no_encoding-kopi.csv')

    # Perform chi-squared test for specific sport and injury
    contingency_table1 = pd.crosstab(df['specific_sport'], df['injury_present'])
    chi2, p1, dof, expected1 = chi2_contingency(contingency_table1)

    # Perform chi-squared test for age and specific sport
    contingency_table2 = pd.crosstab(df['age'], df['specific_sport'])
    chi2, p2, dof, expected2 = chi2_contingency(contingency_table2)

    # Perform chi-squared test for age and frequency of training
    contingency_table3 = pd.crosstab(df['age'], df['frequency_of_training'])
    chi2, p3, dof, expected3 = chi2_contingency(contingency_table3)

    # Create an Explore page
    st.title(':blue[Explore Page] by _Joseph Margaryan_ and _Toba Krubally_ :sunglasses:')

    # Perform chi-squared tests
    st.write("Chi-squared test for Specific Sport vs Injury:")
    st.write("Chi2 = {0:.2f}, p-value = {1}".format(chi2, p1))

    st.write("Chi-squared test for Age vs Specific Sport:")
    st.write("Chi2 = {0:.2f}, p-value = {1}".format(chi2, p2))

    st.write("Chi-squared test for Age vs Frequency of Training:")
    st.write("Chi2 = {0:.2f}, p-value = {1}".format(chi2, p3))

    # Create a heatmap of the contingency table for specific sport and injury
    plt.figure(figsize=(8, 6))
    sns.heatmap(contingency_table1, annot=True, cmap="YlGnBu")
    plt.title("Contingency Table for Specific Sport and Injury")
    plt.xlabel("Injury Present")
    plt.ylabel("Specific Sport")
    st.pyplot(plt)

    # Create a heatmap of the expected frequencies for specific sport and injury
    plt.figure(figsize=(8, 6))
    sns.heatmap(expected1, annot=True, cmap="YlGnBu")
    plt.title("Expected Frequencies for Specific Sport and Injury")
    plt.xlabel("Injury Present")
    plt.ylabel("Specific Sport")
    st.pyplot(plt)

    # Create bar plots of the contingency table for specific sport and injury
    contingency_table1.plot(kind='bar', stacked=True)
    plt.xlabel('Specific Sport', fontsize=10)
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, fontsize=8)
    plt.title('Contingency Table: Specific Sport vs Injury')
    st.pyplot(plt)

    # Create bar plots of the contingency table for age and specific sport
    contingency_table2.plot(kind='bar', stacked=True)
    plt.xlabel('Age', fontsize=10)
    plt.ylabel('Frequency')
    plt.title('Contingency Table: Age vs Specific Sport')
    st.pyplot(plt)

    # Create bar plots of the contingency table for age and frequency of training
    contingency_table3.plot(kind='bar', stacked=True)
    plt.xlabel('Age', fontsize=15)
    plt.ylabel('Frequency')
    plt.title('Contingency Table: Age vs Frequency of Training')
    st.pyplot(plt)

    # group by specific sport and injury_present to get count of injuries
    grouped = df.groupby(['specific_sport', 'injury_present'])['gender'].count().reset_index()

    # pivot to get counts for each sport
    pivoted = pd.pivot_table(grouped, values='gender', index='specific_sport', columns='injury_present', fill_value=0)

    # calculate ratio of positive cases
    pivoted['ratio'] = pivoted[1] / (pivoted[0] + pivoted[1])

    # calculate mean ratio across all sports
    mean_ratio = pivoted['ratio'].mean()

    # add comparison to mean ratio
    pivoted['ratio_diff'] = pivoted['ratio'] - mean_ratio

    # sort by ratio_diff
    pivoted = pivoted.sort_values('ratio_diff')

    # calculate total count of respondents for each sport
    total_count = df.groupby('specific_sport')['gender'].count()

    # add column to pivoted DataFrame
    pivoted['total_count'] = total_count

    # reorder columns
    pivoted = pivoted[['total_count', 0, 1, 'ratio', 'ratio_diff']]

    # rename columns
    pivoted = pivoted.rename(columns={0: 'no injury', 1: 'injury'})

    # Sort our table with ratio_diff descending
    pivoted.sort_values(by='ratio_diff', axis=0, ascending=False, inplace=True)

    # print table
    st.write(pivoted)

    # create the bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    pivoted['ratio'].plot(kind='bar', color='C0')
    plt.axhline(mean_ratio, color='red', linestyle='--')
    plt.ylabel('Ratio of positive cases')
    plt.xlabel('Sport')
    plt.title('Ratio of positive injury cases by sport')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    st.pyplot(plt)

    # Calculate the average percent of injuries in each sport
    sport_injury_ratio = df.groupby('specific_sport')['injury_present'].mean()

    # Calculate the overall mean injury ratio
    overall_mean = df['injury_present'].mean()

    # Calculate the percent difference between the average injury ratios
    difference = sport_injury_ratio - overall_mean

    # Create the bar chart
    difference.plot(kind='bar')

    # Add labels and title
    plt.xlabel('Sports')
    plt.ylabel('Percent Difference')
    plt.xticks(rotation=45, minor=True)
    plt.title('Diff between Average % of Injuries in Sports and Overall Mean Injury Ratio')

    # Show the plot
    st.pyplot(plt)

    # Calculate the percentage of respondents for each sport
    sport_percentages = df.groupby('specific_sport')['injury_present'].count() / df['injury_present'].count()

    # Create the pie chart
    sport_percentages.plot(kind='pie', autopct='%.2f')

    # Add title and labels
    plt.title('Sport Representation % of Respondents')
    plt.ylabel('')

    # Show the plot
    st.pyplot(plt)

    # calculate the proportion of 'Kvinde' and 'Mand' with an injury
    kvinde_injury = df[df['gender'] == 1]['injury_present'].sum()
    mand_injury = df[df['gender'] == 0]['injury_present'].sum()

    # calculate the total number of respondents
    total = df.shape[0]

    # calculate the proportion of 'Kvinde' and 'Mand' with an injury
    kvinde_injury_proportion = kvinde_injury / total
    mand_injury_proportion = mand_injury / total

    st.write('Proportion of Women with an injury:', kvinde_injury_proportion)
    st.write('Proportion of Men with an injury:', mand_injury_proportion)

    # create the bar chart
    plt.bar(x=['Kvinde', 'Mand'], height=[kvinde_injury_proportion, mand_injury_proportion])

    # set the x-axis label
    plt.xlabel('Gender')

    # set the y-axis label
    plt.ylabel('Proportion with Injury')

    # set the title
    plt.title('Proportion of women and men with Injury')

    # show the plot
    st.pyplot(plt)

def page2():
    # Load the data
    df = pd.read_csv('csv_for_app-kopi.csv')

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns='injury_present', axis=0).values,
        df['injury_present'].values,
        test_size=0.2,  # Use 20% of the data for testing
        random_state=42  # Set the random seed for reproducibility
    )

    # Define the KNN model
    kn = KNeighborsClassifier()
    kn.fit(X_train, y_train)

    # Define the SVM model
    svm = SVC(probability=True)  # Use probability estimates
    svm.fit(X_train, y_train)

    # Define the logistic regression model
    lr = LogisticRegression()
    lr.fit(X_train, y_train)

    # Define the decision tree model
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)

    # Compute the accuracy scores
    kn_accuracy = accuracy_score(y_test, kn.predict(X_test))
    svm_accuracy = accuracy_score(y_test, svm.predict(X_test))
    lr_accuracy = accuracy_score(y_test, lr.predict(X_test))
    dt_accuracy = accuracy_score(y_test, dt.predict(X_test))

    # Define the page title
    st.title('Injury Prediction App')

    # Define the select boxes for age, gender, and sport
    age_options = {
        '18-25': 1,
        '26-35': 2,
        '36-45': 3,
        '46+': 4
    }
    age = st.selectbox('Select your age:', options=list(age_options.keys()))

    gender_options = {
        'Man': 0,
        'Woman': 1
    }
    gender = st.selectbox('Select your gender:', options=list(gender_options.keys()))

    sport_options = {
        'BJJ': 'specific_sport_BJJ',
        'Boksning': 'specific_sport_BOKSNING',
        'Bouldering': 'specific_sport_BOULDERING',
        'Dans': 'specific_sport_DANS',
        'Fodbold': 'specific_sport_FODBOLD',
        'MMA': 'specific_sport_MMA',
        'Muay Thai': 'specific_sport_MUAY_THAI',
        'Ridning': 'specific_sport_RIDNING',
        'Svømning': 'specific_sport_SVØMNING'
    }
    sport = st.selectbox('Select your sport:', options=list(sport_options.keys()))

    # Define the function for computing the prediction and accuracy
    def predict_injury(age, gender, sport, model):
        # Create a new row with the selected features
        new_row = {
            'age': age_options[age],
            'gender': gender_options[gender],
            'injury_present': 0  # This will be ignored by the model
        }
        for key, value in sport_options.items():
            new_row[value] = value == sport_options[sport]

            # Predict the injury status using the model
        if model == 'KNN':
            prediction = kn.predict([list(new_row.values())])[0]
        elif model == 'SVM':
            prediction = svm.predict([list(new_row.values())])[0]
        elif model == 'Decision Tree':
            prediction = dt.predict([list(new_row.values())])[0]
        else:  # Use the logistic regression model
            prediction = lr.predict([list(new_row.values())])[0]

        if prediction == 0:
            return 'No injury'
        else:
            return 'Injury'

    # Define the select box for the model
    model_options = {
        'KNN': kn,
        'SVM': svm,
        'Logistic Regression': lr,
        'Decision Tree': dt,
    }
    model = st.selectbox('Select a model:', options=list(model_options.keys()))

    # Add the logistic regression accuracy score to the page
    st.write(f'Logistic Regression accuracy: {lr_accuracy:.2f}')
    st.write(f'KNeighbors Classifier Regression Accuracy: {kn_accuracy:.2f}')
    st.write(f'Decision Tree Classifier Regression Accuracy: {dt_accuracy:.2f}')
    st.write(f'Support Vector Machine Regression Accuracy: {svm_accuracy:.2f}')

    # Add a button to predict the injury status
    if st.button('Predict'):
        # Compute the prediction using the selected model
        prediction = predict_injury(age, gender, sport, model_options[model])
        st.write(f'Prediction: {prediction}')


def app():
    # Create a dropdown menu for selecting the page
    page_options = ["Page 1", "Page 2"]
    page_selection = st.sidebar.selectbox("Select a page", page_options)

    # Show the selected page
    if page_selection == "Page 1":
        page1()
    elif page_selection == "Page 2":
        page2()

if __name__ == "__main__":
    app()












