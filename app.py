
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import pandas as pd
import streamlit as st
import keras
import tensorflow as tf
from keras_facenet import FaceNet
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.neighbors import KNeighborsClassifier

from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px
from htbuilder.funcs import rgba, rgb



# title of the app
title_text = "TracktIt AI"

st.markdown(f"<h1 style='text-align: center;'>{title_text}</h1>", unsafe_allow_html=True)
st.markdown("***")

# control panel
st.sidebar.markdown(f"<h1 style='text-align: center;'>{'Control Panel'}</h1>", unsafe_allow_html=True)
st.sidebar.markdown("***")





# save the variables in st.session_state
if 'facenet' not in st.session_state:
    facenet = FaceNet()
    st.session_state['facenet'] = facenet

# define the harcascade classfier
if 'face_detector' not in st.session_state:
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    st.session_state['face_detector'] = face_detector

if 'encodings' not in st.session_state:
    st.session_state['encodings'] = []

if 'names' not in st.session_state:
    st.session_state['names'] = []

# data of excel sheet 
if 'df' not in st.session_state:
    st.session_state['df'] = pd.DataFrame(columns = ['On_bord_Date','On_bord_Time', 'Attendence_Status',
                                                     'Attendence_Date','Attendence_Time'])
    


# date time function
def current_time():
    return datetime.now().strftime("%H:%M:%S")

def current_date():
    return datetime.now().strftime("%Y-%m-%d")





# select box for choose method to add new user
method = st.sidebar.selectbox("Choose Method to Add New User", ("Camera", "Upload Image"), index = None, placeholder= 'Choose Method',)


if method:

    if method == 'Upload Image':

        # user upload the file 
        upload_file = st.sidebar.file_uploader("Upload Images Only", type=["png", "jpg", "jpeg"], accept_multiple_files=True)


        if st.sidebar.button('Start Training',type = 'primary'):
            if len(upload_file) > 0:

                # define the facenet and detector mode again
                facenet = st.session_state['facenet']
                face_detector = st.session_state['face_detector']

                with st.spinner('Training in progress. Please wait.'):
                    try:
                        for file in upload_file:

                            img = plt.imread(file)
                            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                            faces = face_detector.detectMultiScale(gray, 1.1, minNeighbors= 4, minSize=(10,10))

                            if len(faces) > 0:
                                x, y, w, h = faces[0]
                                x2 = x+w+3
                                y2 = y+h+3
                                face = img[y:y2, x:x2,:].copy()

                                # preprocess face data
                                face_resize = cv2.resize(face, (160, 160), interpolation=cv2.INTER_AREA)
                                face_exdim = np.expand_dims(face_resize, axis=0)

                                #extrate the encoding
                                encode = facenet.embeddings(face_exdim)
                                p_name = file.name.split('.')[0]

                                # save at session state
                                st.session_state['encodings'].append(encode[0])
                                st.session_state['names'].append(p_name)

                                # add data in excel sheet df
                                if not any(st.session_state['df'].index.isin([p_name])):
                                    st.session_state['df'].loc[p_name] = [current_date(), current_time(), 'Not Present',
                                                                        00.00, 00.00]
                                

                        # train the knn model
                        le = LabelEncoder()
                        st.session_state['labels'] = le.fit_transform(st.session_state['names'])
                        st.session_state['label_encoder'] = le
                    
                        knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
                        knn.fit(normalize(np.asarray(st.session_state['encodings'])), st.session_state['labels'])
                        st.session_state['knn'] = knn
            
                    except Exception as e:
                        st.write(f"Each picture must follow a certain format, and its filename should be the person's name.. Error: {e}")


                st.success('Training Completed', icon = '✅')


            else:
                st.warning('Please... upload an image first', icon='⚠️')




    # method camers to add new user
    if method == 'Camera':


        # user name
        user_name = st.sidebar.text_input('User Name', placeholder= 'Enter your name')
    

        # define session state for the buttons
        if 'camera_button' not in st.session_state:
            st.session_state['camera_button'] = False

        if 'capture_button' not in st.session_state:
            st.session_state['capture_button'] = False
        
        if 'end_button' not in st.session_state:
            st.session_state['end_button'] = False



        # camera turn on
        if st.sidebar.button('Camera'):

            if user_name != '':
                st.session_state['camera_button'] = True
                st.success('Camera is enabled.', icon = '✅')
            else:
                st.warning('Please... enter your name first', icon='⚠️')



        # capture button
        if st.sidebar.button('Capture'):

            if not st.session_state['capture_button']:

                if st.session_state['camera_button']:
                    with st.spinner('Please wait....'):
                        time.sleep(2)
                        st.session_state['capture_button'] = True
                        st.success('Capturing images is enabled.', icon = '✅')
                else:
                    st.warning('Please... start the camera first', icon='⚠️')
            else:
                st.warning('Capturing images is already enabled.', icon='⚠️')


        # stop button
        if st.sidebar.button('Stop',type = 'primary'):

            if st.session_state['camera_button']:
                try:
                    with st.spinner('Turning off camera. Please wait....'):

                        time.sleep(2)
                        st.session_state['camera_button'] = False
                        st.session_state['capture_button'] = False

                        # train the knn model
                        le = LabelEncoder()
                        st.session_state['labels'] = le.fit_transform(st.session_state['names'])
                        st.session_state['label_encoder'] = le
                        knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
                        knn.fit(normalize(np.asarray(st.session_state['encodings'])), st.session_state['labels'])
                        st.session_state['knn'] = knn
                        st.success('Training Completed & Camera Closed', icon = '✅')

                except Exception as e:
                    st.warning(f"Camera Closed", icon='✅')
                
            else:
                st.warning('Please... start the camera first', icon='⚠️')
                



        if st.session_state['camera_button']:
            with st.spinner('Turning on camera. Please wait....'):

                # define the web cam
                cap = cv2.VideoCapture(0)

                # define the facenet and detector mode again
                facenet = st.session_state['facenet']
                face_detector = st.session_state['face_detector']

                # define button and placeholder
                frame_placeholder_0 = st.empty()
                time.sleep(1)


            if user_name != '':

                # video 
                a = 1
                while st.session_state['camera_button']:

                    rat, frame = cap.read()

                    if not rat:
                        st.write("Video capture has ended. To restart, click the Camera button.")
                        break
                    
                    # capture button

                    if st.session_state['capture_button']:
                        # find the face in the image
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = face_detector.detectMultiScale(gray, 1.1, minNeighbors= 4, minSize=(10,10))

                        if len(faces) > 0:
                            x, y, w, h = faces[0]
                            x2 = x+w+3
                            y2 = y+h+3
                            face_new = frame[y:y2, x:x2,::-1].copy()

                            # preprocess face data
                            face_resize = cv2.resize(face_new, (160, 160), interpolation=cv2.INTER_AREA)
                            face_exdim = np.expand_dims(face_resize, axis=0)

                            #extrate the encoding
                            encode = facenet.embeddings(face_exdim)

                            # save at session state
                            st.session_state['encodings'].append(encode[0])
                            st.session_state['names'].append(user_name)

                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            cv2.putText(frame, f"number of images captured: {a}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                            # add data in excel sheet df 
                            if not any(st.session_state['df'].index.isin([user_name])):
                                st.session_state['df'].loc[user_name] = [current_date(), current_time(), 'Not Present',
                                                                        00.00, 00.00]

                            a += 1

                    frame_placeholder_0.image(frame, channels='BGR')

            else:
                st.warning('Please... enter your name first', icon='⚠️')





# line in between 2 sactions 
    st.sidebar.markdown('***')

    # describe the  detials to the user
    lines = f"Total {len(st.session_state['encodings'])} images Data In the Model."

    def stream_data(Line =  lines):    
        for word in Line.split():
            yield word + " "
            time.sleep(0.07)


    # button to show and delete data 
    if st.sidebar.button('Show Data'):
        # try:
            with st.spinner('Loading data. Please wait....'):
                time.sleep(2)
                st.header('Attendence Data')
                st.write(st.session_state.get('df'))
                st.header('Data Info')
                st.write(pd.Series(st.session_state.get('names')).value_counts())
                st.write_stream(stream_data())

        # except Exception as e:
        #     st.write(f"Error: {e}")

    if st.sidebar.button('Delete Data'):
        st.session_state['encodings'] = []
        st.session_state['names'] = []
        st.session_state['labels'] = []
        st.session_state['label_encoder'] = None
        st.session_state['knn'] = None
        st.session_state['df'] = st.session_state['df'].drop(index = st.session_state['df'].index)
        st.success('Training data deleted', icon = '✅')

    
    # line in between 2 sactions 
    st.sidebar.markdown('***')
    st.sidebar.markdown('**Application Start or End Controllers**')
        



    ##-----------------------  start the web cam using open cv --------------------------------------------------



    # place_holder
    frame_placeholder = st.empty()


    if 'Streaming' not in st.session_state:
        st.session_state.Streaming = False


    if st.sidebar.button('Start', type='primary'):
        with st.spinner("We won't keep you waiting long."):
            st.session_state.Streaming = True
            cap = cv2.VideoCapture(0)


    if st.sidebar.button('End', type='primary'):
        if st.session_state.Streaming:
            with st.spinner("Turning off camera. Please wait...."):
                time.sleep(2)
                st.session_state.Streaming = False
        else:
            st.warning("Please... start the Webcam first", icon="⚠️")


    if st.session_state.Streaming:

        try:

            if len(st.session_state['names']) >= 5:

                # define the important variables 
                face_detector = st.session_state['face_detector']
                facenet = st.session_state['facenet']
                knn = st.session_state['knn']
                le = st.session_state['label_encoder']

                while st.session_state.Streaming:
                    rat, frame = cap.read()

                    if not rat:
                        st.write("Video capture has ended. To restart, click the Start button.")
                        break

                    # find the face in the image
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_detector.detectMultiScale(gray, 1.1, minNeighbors= 4, minSize=(5, 5))

                    if len(faces) > 0:
                        for face in faces:
                            x, y, w, h = face
                            x2 = x+w+3
                            y2 = y+h+3
                            face = rgb_frame[y:y2, x:x2,:].copy()

                            # resize the face
                            face_resize = cv2.resize(face, (160, 160), interpolation=cv2.INTER_AREA)
                        
                            # expend dim
                            face_exdim = np.expand_dims(face_resize, axis=0)

                            # encode the face
                            encode = facenet.embeddings(face_exdim)

                            # predict
                            pred = knn.predict(normalize(encode))
                            dist, ind = knn.kneighbors(normalize(encode), n_neighbors=5, return_distance=True)
                            avg_dist = np.mean(dist, axis=1)

                            if avg_dist[0] <= 0.8:
                                name = le.inverse_transform(pred)[0]
                                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                                cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                                # mark attendence 
                                if st.session_state['df'].loc[name,'Attendence_Status'] == 'Not Present':
                                    st.session_state['df'].loc[name,'Attendence_Status'] = 'Present'
                                    st.session_state['df'].loc[name,'Attendence_Date'] = current_date()
                                    st.session_state['df'].loc[name,'Attendence_Time'] = current_time()

                            else:
                                name = 'Unknown'
                                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)
                                cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)


                    frame_placeholder.image(frame, channels='BGR')

            else:
                st.warning('At least 5 Images Are Required ', icon='⚠️')
                st.info('Press End Button and Start the WebCam Again with At Least 5 Images', icon='💡')

        except Exception as e:
            st.warning(f"oops! Something went wrong... Press the End button And Start Again.")




    st.sidebar.markdown("***")

    #download csv button
    st.sidebar.download_button(label = 'Download CSV', data = st.session_state['df'].to_csv(), 
                            file_name = 'Attendence.csv', mime = 'text/csv')

            
else:
        
    # describe the  detials to the user

    st.header("Overview")
    st.write("""
    TrackIt AI is an advanced attendance system that uses facial recognition to mark attendance. 
    It stores the attendance data in an Excel sheet that users can download. This system is user-friendly 
    and offers two methods for adding image data: using a camera or uploading files.
    """)

    st.header("How to Use TrackIt AI")

    st.subheader("1. Adding Image Data")
    st.write("*Using Camera:*")
    st.write("""
    - Enter your name in the provided field.
    - Click the *Camera* button to start the camera.
    - Click the *Capture* button to take pictures. The application will automatically capture the images.
    - Click the *Stop* button to stop the camera. The captured images will be used to train the model.
    - Click the *Show* button to view your data and info.
    - Click the *Delete* button to remove all your image data.
    """)

    st.write("*Uploading Files:*")
    st.write("""
    - Ensure that each image file name is the person’s name followed by a dot.
    - Upload the images from your local files.
    - For best performance, add at least 5 images per user and ensure a balanced number of images for each user.
    - Click the *Start Training* button to train the model. Training will take some time.
    """)

    st.subheader("2. Recording Attendance")
    st.write("""
    - After training, click the *Start* button to start the camera.
    - The system will detect and recognize the faces of registered users and mark their attendance in an Excel sheet.
    - Unknown users will also be detected but not marked.
    - Click the *End* button to stop the camera.
    - Click the *Show Data* button to view the data of the Excel sheet.
    - Click the *Download CSV* button to download the attendance data as a CSV file.
    """)

    st.info("""Note:- If the model takes time to detect your face, please clean your webcam, adjust your pose, ensure you're 
            not too far from the camera,and check that the lighting is adequate. Make sure your face is clearly visible.""", icon=':material/info:')
    
    st.write("""By following these steps, users can effectively utilize the TrackIt AI Attendance System to manage attendance through 
             facial recognition.""")






# footer 

try:

    def image(src_as_string, **style):
        return img(src=src_as_string, style=styles(**style))


    def link(link, text, **style):
        return a(_href=link, _target="_blank", style=styles(**style))(text)


    def layout(*args):

        style = """
        <style>
        # MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stApp { bottom: 5px; }
        </style>
        """

        style_div = styles(
            position="fixed",
            left=0,
            bottom=0,
            margin=px(0, 0, 0, 150),
            width=percent(100),
            color="black",
            text_align="center",
            height=px(105),
            opacity=1
        )

        style_hr = styles(
            display="block",
            margin=px(8, 8, "auto", "auto"),
            border_style="inset",
            border_width=px(0)
        )

        body = p()
        foot = div(
            style=style_div
        )(
            hr(
                style=style_hr
            ),
            body
        )

        st.markdown(style, unsafe_allow_html=True)

        for arg in args:
            if isinstance(arg, str):
                body(arg)

            elif isinstance(arg, HtmlElement):
                body(arg)

        st.markdown(str(foot), unsafe_allow_html=True)


    def footer():
        myargs = [
            "Made in ",
            image('https://avatars3.githubusercontent.com/u/45109972?s=400&v=4',
                width=px(25), height=px(25)),
            " with ❤️ by ",
            link("https://github.com/Mohitjhedu", "@mohit_jhedu"),]
        layout(*myargs)


    if __name__ == "__main__":
        footer()


except Exception as e:
    print(".")


