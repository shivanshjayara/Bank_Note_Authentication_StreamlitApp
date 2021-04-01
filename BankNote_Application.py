import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
from sklearn.model_selection import train_test_split
import plotly

file = open('classifier.pkl', 'rb')
classifier = pickle.load(file)

header=st.beta_container()
data_set=st.beta_container()
features=st.beta_container()
model_training=st.beta_container()


st.markdown(
    '''
    <style>
    .main {
    color: #ffff5;
    background-color:yellow;
    }
    </style>
    ''',
    unsafe_allow_html=True
)


@st.cache
def get_data(data1,data2):
    org_data = pd.read_csv(data1)  # Original Data set
    trans_data = pd.read_csv(data2)  # Data set after transformation using StandardScaler
    return org_data, trans_data


with header:
    st.title('Bank Note Authentication')
    st.text('We will see whether a person will be given a loan or not')


with data_set:
    st.header('Bank authentication model:')
    data,std_data=get_data('data_set.csv','banknote_dataXY.csv')

    st.write(data.head(50))
    st.text('Above are the original data (Displaying only 50 records')


    st.title('Bar chart of Standardised Data set')
    st.subheader('**Variance**')
    st.bar_chart(std_data['variance'].head(50))

    st.subheader('**Skewness**')
    st.bar_chart(std_data['skewness'].head(50))

    st.subheader('**Curtosis**')
    st.bar_chart(std_data['curtosis'].head(50))

    st.subheader('**Entropy**')
    st.bar_chart(std_data['entropy'].head(50))



    st.title('Distribution Plots for given features of Standardised Data')

    st.subheader('1) Variance')
    variance_plot = [std_data['variance']]
    fig_var = ff.create_distplot(hist_data=variance_plot, group_labels=['variance'])
    st.plotly_chart(fig_var, use_container_width=True)

    st.subheader('2) Skewness')
    skewness_plot = [std_data['skewness']]
    fig_skewness = ff.create_distplot(hist_data=skewness_plot, group_labels=['skewness'])
    st.plotly_chart(fig_skewness, use_container_width=True)

    st.subheader('3) Curtosis')
    curtosis_plot = [std_data['curtosis']]
    fig_curtosis = ff.create_distplot(hist_data=curtosis_plot, group_labels=['curtosis'])
    st.plotly_chart(fig_curtosis, use_container_width=True)

    st.subheader('4) Entropy')
    entropy_plot = [std_data['entropy']]
    fig_entropy = ff.create_distplot(hist_data=entropy_plot, group_labels=['entropy'])
    st.plotly_chart(fig_entropy, use_container_width=True)



with features:
    st.header('Here are the features')
    st.write(std_data.columns)

with model_training:
    st.header('Here the model is getting train')
    # st.text(list(data.columns[:-1]))   # displaying features name in the form of list
    st.text('')
    st.text('')

    #creating a columns
    col1, col2, col3, col4, col5= st.beta_columns(5)

    #creating slider
    st.text('')
    # depth= col.slider('Maximum Depth', min_value= 10, max_value=100, value= 20, step=10)
    # estimator= col.selectbox('How many trees you wan to use in this?', options=[100,200,300,400,500, 'no limit'], index=0)

    input_feature_1=col1.text_input('Variance')
    input_feature_2=col1.text_input('Skewness')
    input_feature_3=col1.text_input('Curtosis')
    input_feature_4=col1.text_input('Entropy')

    result=''
    if col1.button('Predict'):
        result=classifier.predict([[input_feature_1,input_feature_2,input_feature_3,input_feature_4]])

    st.success(f'The output is : {result}')
    st.text('0- Not eligible for loan            1-Eligible for loan')

    x = std_data.drop(columns='class')
    y = std_data[['class']]

    x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.33,random_state=101)

    dt_model= DecisionTreeClassifier()
    dt_model.fit(x_train,y_train)
    predict_y=dt_model.predict(x_test)

    col3.subheader('Accuracy score: ')
    col3.write(f'The accuracy of the model w.r.t. Decision Tree classifier:\n {round(accuracy_score(y_test,predict_y),3)*100} %')
    #
    # col5.subheader('Confusion matrix: ')
    # col5.write(confusion_matrix(y_test,predict_y))
