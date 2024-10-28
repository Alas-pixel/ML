import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

def main():
    
    st.set_page_config(
        page_title="Machine Learning/Perdict",
        page_icon="balloon:"
    )
    st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://wallpaperset.com/w/full/d/0/5/248958.jpg');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }
    </style>
    """,
    unsafe_allow_html=True
    )
    st.markdown(
    """
    <style>
    .shadow-text {
        # text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
        color: black;  /* Text color */
    }
    </style>
    """,
    unsafe_allow_html=True
)
    st.markdown('<h1 class="shadow-text">Hello, Streamlit!</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="shadow-text">Business Prediction/ Car Purchasing Machine Learning algorithms</h2>', unsafe_allow_html=True)
    st.markdown('<h3 class="shadow-text">Using Streamlit App / Supervised Learning</h3>', unsafe_allow_html=True)
    st.image("2441903.jpg", use_column_width=True)
    algorithm = st.selectbox(":blue[Algorithms]", ["KNN", "SVM", "Naive Bayes", "Decision Tree", "Random Forest"])
    uploadfile = st.file_uploader(":blue[Please Upload a file]")
        
    if uploadfile is not None:
        df = pd.read_csv(uploadfile)
        st.write(df)
        df.drop(columns=['User ID', 'Gender'], inplace=True) 
        gen=st.button(":blue[<generate>]")
        
        x = df.iloc[:, [0, 1]].values
        y = df.iloc[:, 2].values
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        if gen:
            if algorithm == "KNN":
                classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
                classifier.fit(x_train, y_train)
                y_pred = classifier.predict(x_test)
                st.write(":blue[Confusion Matrix:]")
                cm =confusion_matrix(y_test, y_pred)
                print(cm)
                st.write(cm)
                st.write(":blue[Accuracy:]", accuracy_score(y_test, y_pred))
                st.write(":blue[Classification Report:]")
                st.write(classification_report(y_test, y_pred))
                st.write(":blue[Visualization:]")

                col1, col2 = st.columns(2)
                with col1:
                    st.write(":blue[Test Data]")
                    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test)
                    plt.xlabel("Feature 1")
                    plt.ylabel("Feature 2")
                    plt.title("Test Data Set")
                    st.pyplot(plt)
                with col2:    
                    st.write(":blue[Train Data]")
                    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
                    plt.xlabel("Feature 1")
                    plt.ylabel("Feature 2")
                    plt.title("Train Data Set")
                    st.pyplot(plt)
                
        
            elif algorithm == "SVM":
                classifier=SVC(kernel='linear',random_state=0)
                classifier.fit(x_train,y_train)
                y_pred=classifier.predict(x_test)
                st.write(":blue[Confusion Matrix:]")
                cm=confusion_matrix(y_test,y_pred)
                print(cm)
                st.write(cm)
                st.write(":blue[Accuracy:]", accuracy_score(y_test, y_pred))
                st.write(":blue[Classification Report:]")
                st.write(classification_report(y_test, y_pred))
                st.write(":blue[Visualization:]")
            

                col1, col2 = st.columns(2)
                with col1:
                    st.write(":blue[Test Data]")
                    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test)
                    plt.xlabel("Feature 1")
                    plt.ylabel("Feature 2")
                    plt.title("Test Data Set")
                    st.pyplot(plt)
                with col2:    
                    st.write(":blue[Train Data]")
                    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
                    plt.xlabel("Feature 1")
                    plt.ylabel("Feature 2")
                    plt.title("Train Data Set")
                    st.pyplot(plt)

            elif algorithm == "Naive Bayes":
                classifier = GaussianNB()
                classifier.fit(x_train, y_train)
                y_pred = classifier.predict(x_test)
                st.write(":blue[Confusion Matrix:]")
                cm = confusion_matrix(y_test, y_pred)
                print(cm)
                st.write(cm)
                st.write(":blue[Accuracy:]", accuracy_score(y_test, y_pred))
                st.write(":blue[Classification Report:]")
                st.write(classification_report(y_test, y_pred))
                st.write(":blue[Visualization:]")

                col1, col2 = st.columns(2)
                with col1:
                    st.write(":blue[Test Data]")
                    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test)
                    plt.xlabel("Feature 1")
                    plt.ylabel("Feature 2")
                    plt.title("Test Data Set")
                    st.pyplot(plt)
                with col2:    
                    st.write(":blue[Train Data]")
                    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
                    plt.xlabel("Feature 1")
                    plt.ylabel("Feature 2")
                    plt.title("Train Data Set")
                    st.pyplot(plt)

            elif algorithm == "Decision Tree":
                classifier = DecisionTreeClassifier(random_state=0)
                classifier.fit(x_train, y_train)
                y_pred = classifier.predict(x_test)
                st.write(":blue[Confusion Matrix:]")
                cm = confusion_matrix(y_test, y_pred)
                print(cm)
                st.write(cm)
                st.write(":blue[Accuracy:]", accuracy_score(y_test, y_pred))
                st.write(":blue[Classification Report:]")
                st.write(classification_report(y_test, y_pred))
                st.write(":blue[Visualization:]")
            
                col1, col2 = st.columns(2)
                with col1:
                    st.write(":blue[Test Data]")
                    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test)
                    plt.xlabel("Feature 1")
                    plt.ylabel("Feature 2")
                    plt.title("Test Data Set")
                    st.pyplot(plt)    
                with col2:    
                    st.write(":blue[Train Data]")
                    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
                    plt.xlabel("Feature 1")
                    plt.ylabel("Feature 2")
                    plt.title("Train Data Set")
                    st.pyplot(plt)
            
            elif algorithm == "Random Forest":
                classifier = RandomForestClassifier(random_state=0)
                classifier.fit(x_train, y_train)
                y_pred = classifier.predict(x_test)
                st.write("Confusion Matrix:")
                cm = confusion_matrix(y_test, y_pred)
                print(cm)
                st.write(cm)
                st.write(":blue[Accuracy:]", accuracy_score(y_test, y_pred))
                st.write(":blue[Classification Report:]")
                st.write(classification_report(y_test, y_pred))
                st.write(":blue[Visualization:]")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(":blue[Test Data]")
                    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test)
                    plt.xlabel("Feature 1")
                    plt.ylabel("Feature 2")
                    plt.title("Test Data Set")
                    st.pyplot(plt)
                with col2:    
                    st.write(":blue[Train Data]")
                    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
                    plt.xlabel("Feature 1")
                    plt.ylabel("Feature 2")
                    plt.title("Train Data Set")
                    st.pyplot(plt)
        
            
            
        
   
if __name__=="__main__":
    main()