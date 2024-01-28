import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from matplotlib import pyplot as plt
import pickle
from sklearn import tree
import streamlit as st
import numpy
from PIL import Image
import plotly.express as px
from pyDOE import *
from itertools import product
import itertools
import csv
import random
import numpy as np
import itertools
import csv
import random

def DoE():
    # Define factors and their levels
    factors = {'Factor1': [0, 1], 'Factor2': [0, 1], 'Factor3': [0, 1],'Factor4': [0, 1]}

    # Generate all combinations of factor levels
    experiment_design = list(itertools.product(*[factors[key] for key in factors]))

    # Create a CSV file to save the experiment design
    csv_file_path = 'experiment_design_with_zero_target.csv'

    # Write header to CSV file
    header = list(factors.keys()) + ['Target']
    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(header)

    # Append experiment design with zero target to CSV file
    with open(csv_file_path, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        for row in experiment_design:
            row = list(row)  # Convert the tuple to a list
            row += [random.randint(0, 10)]
            csv_writer.writerow(row)

    print(f"Experiment design with zero target has been saved to {csv_file_path}")

DoE()


#Roh_data = pd.read_csv("C:\\Users\\thaya\\Downloads\\CTQ_Fastline.csv",delimiter=";")
Roh_data = pd.read_csv("CTQ_Fastline.csv",delimiter=";")
print("Row Data: ", Roh_data)
df = pd.DataFrame(Roh_data)
print(df.describe())

# corr= df.corr()['Gender']
# print(corr)

feature_column = ['EMPB', 'Medizin', 'Luftfahrt', 'QSV', 'Kundennorm', 'Kundenfreigabe Fast Lane', 'Produktkosten']
Xi = Roh_data[feature_column]
target_column = ['Fast Lane']
Y = Roh_data[target_column]
print(Xi)
print(Y)

#X_train, X_test, Y_train, Y_test = train_test_split(Xi, Y, test_size=None, random_state=1)


DCF = DecisionTreeClassifier()
#DCF = DecisionTreeClassifier(criterion="entropy", max_depth=20)
DCF_FIT_Model = DCF.fit(Xi, Y)
#Y_Predict = DCF_FIT_Model.predict(X_test)
# print("Accuracy: ", metrics.accuracy_score(Y_test,Y_Predict))

#fig = plt.figure()
tree.plot_tree(DCF_FIT_Model, feature_names=feature_column,filled=True, rounded=True,)
plt.show()
#['EMPB', 'Medizin', 'Luftfahrt', 'QSV', 'Kundennorm', 'Kundenfreigabe Fast Lane', 'Produktkosten']
Y_single_predictive = DCF_FIT_Model.predict([[0, 0, 0, 0, 0, 0, 0]])
#print("Fast Lange:", Y_single_predictive)
if Y_single_predictive==1:
    print("Fast Lane",Y_single_predictive)
else:
    print("Kein Fast Lane",Y_single_predictive)


def deploy():
    filename = "trained_model_fast_lane.sav"
    pickle.dump(DCF, open(filename, 'wb'))


deploy()

loaded_model = pickle.load(open("trained_model_fast_lane.sav", 'rb'))
#Y_single_predictive1 = loaded_model.predict([[0, 0, 0, 0, 0, 0, 1, ]])
#print("Fast Lange:", Y_single_predictive1)
def xy_prediction(inputs):
    input_data=(inputs)
    input_data_as_nb=numpy.asarray(input_data)
    print(input_data_as_nb)
    Y_single_predictive1 = loaded_model.predict([input_data_as_nb])
    #print("Geschlecht:", Y_single_predictive1)
    if Y_single_predictive1 == 1:
        return 'Fast Lane'
    else:
        return 'Not Fast Lane'


def main():
    st.set_page_config(page_title="Decision Maker")
    st.title("Decision Adviser Fast lane or not Fast lane")
    st.subheader("Please fill the form")
    col1,col2=st.columns(2)
    # ['EMPB', 'Medizin', 'Luftfahrt', 'QSV', 'Kundennorm', 'Kundenfreigabe Fast Lane', 'Produktkosten']
    with col1:

        EMPB = st.selectbox('EMPB:', (0, 1))
        Medizin = st.selectbox('Medizin:', (0, 1))
        Luftfahrt = st.selectbox('Luftfahrt:', (0, 1))
        QSV = st.selectbox('QSV vorhanden:', (0, 1))

    with col2:
        Kundennorm = st.selectbox('Kundennorm vorhanden:', (0, 1))
        Kundenfreigabe_Fast_Lane = st.selectbox('Kundenfreigabe für Fast Lane:', (0, 1))
        Produktkosten = st.selectbox('Produktkosten klein:', (0, 1))
        pred = ""
        if st.button("Make Decision"):
            pred = xy_prediction([EMPB, Medizin, Luftfahrt, QSV, Kundennorm, Kundenfreigabe_Fast_Lane, Produktkosten])

        st.success(pred)


    #show_matrix=st.button("Show Matrix")
    #if show_matrix:
        #data = pd.read_csv("C:\\Users\\thaya\\Downloads\\CTQ_Fastline.csv", delimiter=";")
        #st.dataframe(data)
        #st.line_chart(data)
        #st.bar_chart(data)
        #st.area_chart(data)


    #img= Image.open(r'C:\Users\thaya\OneDrive\Dokumente\Regression.png')
    #st.image(img)
    #video = open(r'C:\Users\thaya\OneDrive\Bilder\Eigene Aufnahmen\WIN_20210620_15_30_13_Pro.mp4','rb')
    #st.video(video,format='video/mp4')




if __name__=='__main__':
    main()