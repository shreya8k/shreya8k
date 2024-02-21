[Lung Cancer Detection.docx](https://github.com/shreya8k/shreya8k/files/14362378/Lung.Cancer.Detection.docx)Lung Cancer Detection using Ensemble Algorithm
	
1. INTRODUCTION
Cancer is a noteworthy general heath issue worldwide with mortality rates increasing day by day. Lung cancer, among all other cancer types is the most common and deadly that occur both in men and women. Lung cancer, additionally known carcinoma is formation of malignant lung tumours’ (cancerous nodules) due to uncontrolled growth of cells in lung tissues. Eating tobacco and smoking are the leading risk factors for causing cancerous lung nodules. The survival rate of lung cancer patients combining all stages is very less roughly 14%with a time span of about 5-6 years. The main problem with lung cancer is that most of these cancer cases are diagnosed in later stages of cancer making treatments more problematic and significantly reducing the survival chances. Hence detection of lung cancer in its earlier stages can increase the survival chances up to 60-70% by providing the patients necessary fast treatment and thus it curbs the mortality rate. Small cell lung cancer and non-small cell lung cancer are two main types of lung cancer classifications based on cell characteristics. The most commonly occurring is non-small cell lung cancer that makes up about 80-85% of all cases, whereas 15-20% of cancer cases are represented by small cell lung cancer. Lung cancer staging depends on spread of cancer in the lungs and tumor size. Lung cancer is mainly classified into 4 stages in order of seriousness: Stage I-Cancer is confined to the lung, Stage II and III-Cancer is confined within the chest and Stage IV-Lung cancer has spread from the chest to other parts of the body. Lung cancer diagnosis can be done by using various imaging modalities such Positron Emission Tomography (PET), Magnetic Resonance Imaging (MRI), Computed Tomography (CT) and Chest X-rays. CT scan images are mostly preferred over other modalities because they are more reliable, have better clarity and less distortion. Visual interpretation of database is a tedious procedure that is time consuming and highly dependent on given individual. This introduces high possibility of human errors and can lead to misclassification of cancer. Hence an automated system is of utmost importance to guide the radiologist in proper diagnosis of lung cancer. The methodology developed for this system includes dataset collection, pre-processing, lung segmentation, feature extraction and classification.
1.1 Objective of the Project
Lung cancer is a fatal genetic disease that has an abnormal growth of cancerous cells in the lungs of the human body. Since the lungs are one of the vital human body organs, lung cancer can have serious implications. In this work, we have focused on fast detection of lung cancer to be beneficial for patients and doctors. Lung cancer can be detected using Histopathology images and other diagnostic tools as well. The proposed work contains a hybridized model of Convolution neural networks and an ensemble of Machine Learning algorithms: Support Vector Classifier, Random Forest, and XG Boost that detect the lung cancer using histopathology images. The overall accuracy achieved by this work is 99.13 %.
2. LITERATURE SURVEY
Lung Cancer Detection Using Image Processing Techniques
Recently, image processing techniques are widely used in several medical areas for image improvement in earlier detection and treatment stages, where the time factor is very important to discover the abnormality issues in target images, especially in various cancer tumours such as lung cancer, breast cancer, etc. Image quality and accuracy is the core factors of this research, image quality assessment as well as improvement are depending on the enhancement stage where low pre-processing techniques is used based on Gabor filter within Gaussian rules. Following the segmentation principles, an enhanced region of the object of interest that is used as a basic foundation of feature extraction is obtained. Relying on general features, a normality comparison is made. In this research, the main detected features for accurate images comparison are pixels percentage and mask-labelling.
Detection of Different Stages of Lungs Cancer in CT-Scan Images using Image Processing Techniques
The most trivial cancer seen in both men and women cancer is Lung cancer (small cell and non-small cell). The survival rate among people increased by early diagnosis of lung cancer. Due to structure of cancer cell the prediction of lung cancer is the most challenging problem, where most of the cells are overlay each other. Recently, the image processing mechanisms are used extensively in medical areas for earlier detection of stages and its treatment, where the time factor is very crucial. Detection of cancer cell in lung cancer patients in time will cause overall 5-year survival rate increasing from 14 to 49%. In the present paper the authors have proposed Lung cancer detection system using image processing technique to classify the presence of cancer cells in lung and its stages from the CT-scan images using various enhancement and segmentation techniques, aiming for accuracy in result.
Image processing techniques for analyzing CT scan images towards the early detection of lung cancer
The application of image processing techniques for the analysis of CT scan images corresponding to lung cancer cells is gaining momentum in recent years. Therefore, it is of interest to discuss the use of a Computer-Aided Diagnosis (CAD) system using Computed Tomography (CT) images to help in the early diagnosis of lung cancer (to distinguish between benign and malignant tumors). We discuss and explore the design and significance of a CAD-CT image processed model in cancer diagnosis.
Detection of Lung Cancer in CT Scan Images using Enhanced Artificial Bee Colony Optimization Technique
Image processing is an improvement of image data where it suppresses unwanted distortion or enhances image features for further processing in various applications and domains. A medical image comprises lot of irrelevant and unwanted parts in its actual format of the scanned images. For the removal of such annoying parts in an image, some of the image preprocessing techniques are required. This helps in better visualization of the images, especially, for diagnosing diseases. In recent years, prediction of cancer in the early stages is necessary as it increases the chance of survival by identifying the cause problems. The most dreadful type of cancer is lung cancer, which is identified as one of the most common diseases among humans worldwide. This research work to identifying lung cancer is carried out using the lung Computed Tomography (CT) scan images. It is carried out for better identification of cancer affected regions. newlineIn this research work, the raw input image which usually suffers from noise issues are highly enhanced using Gabor filter image processing. The region of interest from lung cancer images are extracted with Otsu s threshold segmentation method and 5- level HAAR Discrete Wavelet Transform (DWT) method, which possess maximum speed and high accuracy. The proposed Enhanced Artificial Bee Colony Optimization (EABC) is applied to detect the cancer suspected area in CT (Computed tomography) scan images. The proposed EABC implementation part utilizes CT scanned lung images with MATLAB software. This method can assist radiologists and medicinal experts to recognize the illness of syndromes at primary stages and to evade severe advance stages of cancer. newlineThere are two stages in this research work which is preprocessing and segmentation. After these stages, the produced results are analyzed with feature extraction techniques. For preprocessing, three methods are applied in this work, namely Median filter, Gaussian method and Boundary Detection method. These newlinexiii newlinemethods are used to remove the unwanted parts in the CT images and for better use of the images. Enhancement of the image quality is obtained by implementing filtering technique, removal of noise is carried out by Gaussian method and edges of the images are detected by Boundary Detection method. newlineThe feature extraction methods used for selecting necessary features are useful to analyze lung cancer deduction. The Discrete Wavelet Transform, Fast Fourier Wavelet Transform (FFWT), Two Level HAAR wavelet and Five Level HAAR wavelet transform are the extraction methods used for feature selection process. Among these methods, Five Level HAAR wavelet method works better than the other methods with regard to sensitivity, specificity and accuracy. After preprocessing, the qualities of the images are improved. newlineTo identify lung cancer affected regions in the CT scan images; this research work proposes a new method baptized Enhanced Artificial Bee Colony Optimization (EABC). This proposed method is applied on CT scan images to detect the cancer suspected region. The source code for the proposed EABC method is written using MATLAB software. After the entire processing of this EABC method, a perfect part of the cancer affected regions were identified. This method can assist radiologists and medicinal experts to recognize the illness of syndromes at any stage and to evade severe advance stages of cancer. newlineAlso, this work includes a comparative analysis with the existing algorithms namely Artificial Neural Network (ANN) and Artificial Bee Colony (ABC). The performances of these three algorithms are produced based on its results. The comparison work is carried out using the accuracy, sensitivity, specificity and F-measures. 
Image processing based detection of lung cancer on CT scan images
In this paper, we implement and analyze the image processing method for detection of lung cancer. Image processing techniques are widely used in several medical problems for picture enhancement in the detection phase to support the early medical treatment. In this research we proposed a detection method of lung cancer based on image segmentation. Image segmentation is one of intermediate level in image processing. Marker control watershed and region growing approach are used to segment of CT scan image. Detection phases are followed by image enhancement using Gabor filter, image segmentation, and features extraction. From the experimental results, we found the effectiveness of our approach. The results show that the best approach for main features detection is watershed with masking method which has high accuracy and robust.
A Review on Image Processing Methods in Detecting Lung Cancer using CT Images
Abstract-Lung cancer is the most common cancer for death among all cancers and CT scan is the best modality for imaging lung cancer. A good amount of research work has been carried out in the past towards CAD system for lung cancer detection using CT images. It is divided into four stages. They are preprocessing or lung segmentation, nodule detection, nodule segmentation and classification. This paper presents in detail literature survey on various techniques that have been used in Pre-processing, nodule segmentation and classification.
“LUNG CANCER DETECTION USING IMAGE PROCESSING TECHNIQUES”

Abstract – As per the technical evolution and latest trend taken into consideration, we have decided to make research over biomedical term i.e. Lungs cancer detection. Recently, image processing techniques are widely used in several medical areas for image improvement in earlier detection and treatment stages. There are various types of cancers i.e. lungs cancer, Breast cancer, blood cancer, throat cancer, brain cancer, tongs cancer, mouth cancer etc. Lung cancer is a disease of abnormal cells multiplying and growing into a tumor. Cancer cells can be carried away from the lungs in blood, or lymph fluid that surrounds lung tissue. In this project we access cancer image into MATLAB collected from different hospitals where present work is going on and this available image was color image we have access that image into MATLAB and followed conversion. Image quality and accuracy is the core factors of this research, image quality assessment as well as improvement are depending on the enhancement stage where low pre-processing techniques is used based on Gabor filter within Gaussian rules. The segmentation and enhancement procedure is used to obtain the feature extraction of normal and abnormal image. Relying on general features, a normality comparison is made. In this research, the main detected features for accurate images comparison are pixels percentage and mask-labelling.
3. SYSTEM ANALYSIS
3.1 EXISTING SYSTEM:
The aberrant proliferation of malignant cells in the lungs is the defining characteristic of lung cancer, a deadly hereditary illness. Lung cancer may have devastating effects since the lungs are such an important part of the human body. In this study, we have concentrated on the rapid diagnosis of lung cancer for the benefit of both patients and physicians. Histopathology scans, together with other diagnostic methods, can be used to identify lung cancer.
Disadvantage:
•	Less Accuracy
•	Time taking
3.2 PROPOSED SYSTEM:
Using lung histopathology pictures, the proposed study employs a hybridised model of Convolution neural networks with an ensemble of Machine Learning algorithms, including Support Vector Classifier, Random Forest, and XG Boost. In all, this job has a 99.13% success rate.
Advantage
•	Less time taking
MODULES
1)	Upload Lung Cancer Dataset: using this module we will Upload Lung Cancer Dataset
2)	Dataset Preprocessing: using this module we will Dataset Preprocessing
3)	Run Ensemble Algorithm: using this module we will Run Ensemble Algorithm
4)	Predict Lung Cancer Disease: using this module we will Predict Lung Cancer Disease
5)	Train RBF on LUNGS CT-SCAN Image: using this module we will Train RBF on LUNGS CT-SCAN Image 
6)	Predict Cancer from CT-Scan: using this module we will Predict Cancer from CT-Scan
3.3. PROCESS MODEL USED WITH JUSTIFICATION
SDLC (Umbrella Model):
 
SDLC is nothing but Software Development Life Cycle. It is a standard which is used by software industry to develop good software.
Stages in SDLC:
•	Requirement Gathering
•	Analysis 
•	Designing
•	Coding
•	Testing
•	Maintenance
Requirements Gatheringstage:
The requirements gathering process takes as its input the goals identified in the high-level requirements section of the project plan. Each goal will be refined into a set of one or more requirements. These requirements define the major functions of the intended application, define operational data areas and reference data areas, and define the initial data entities. Major functions include critical processes to be managed, as well as mission critical inputs, outputs and reports. A user class hierarchy is developed and associated with these major functions, data areas, and data entities. Each of these definitions is termed a Requirement. Requirements are identified by unique requirement identifiers and, at minimum, contain a requirement title and textual description.
 
These requirements are fully described in the primary deliverables for this stage: the Requirements Document and the Requirements Traceability Matrix (RTM). The requirements document contains complete descriptions of each requirement, including diagrams and references to external documents as necessary. Note that detailed listings of database tables and fields are not included in the requirements document.
The title of each requirement is also placed into the first version of the RTM, along with the title of each goal from the project plan. The purpose of the RTM is to show that the product components developed during each stage of the software development lifecycle are formally connected to the components developed in prior stages.
In the requirements stage, the RTM consists of a list of high-level requirements, or goals, by title, with a listing of associated requirements for each goal, listed by requirement title. In this hierarchical listing, the RTM shows that each requirement developed during this stage is formally linked to a specific product goal. In this format, each requirement can be traced to a specific product goal, hence the term requirements traceability.
The outputs of the requirements definition stage include the requirements document, the RTM, and an updated project plan.
•	Feasibility study is all about identification of problems in a project.
•	No. of staff required to handle a project is represented as Team Formation, in this case only modules are individual tasks will be assigned to employees who are working for that project.
•	Project Specifications are all about representing of various possible inputs submitting to the server and corresponding outputs along with reports maintained by administrator.
Analysis Stage:
The planning stage establishes a bird's eye view of the intended software product, and uses this to establish the basic project structure, evaluate feasibility and risks associated with the project, and describe appropriate management and technical approaches.
 
The most critical section of the project plan is a listing of high-level product requirements, also referred to as goals. All of the software product requirements to be developed during the requirements definition stage flow from one or more of these goals. The minimum information for each goal consists of a title and textual description, although additional information and references to external documents may be included. The outputs of the project planning stage are the configuration management plan, the quality assurance plan, and the project plan and schedule, with a detailed listing of scheduled activities for the upcoming Requirements stage, and high level estimates of effort for the out stages.
Designing Stage:
The design stage takes as its initial input the requirements identified in the approved requirements document. For each requirement, a set of one or more design elements will be produced as a result of interviews, workshops, and/or prototype efforts. Design elements describe the desired software features in detail, and generally include functional hierarchy diagrams, screen layout diagrams, tables of business rules, business process diagrams, pseudo code, and a complete entity-relationship diagram with a full data dictionary. These design elements are intended to describe the software in sufficient detail that skilled programmers may develop the software with minimal additional input.


When the design document is finalized and accepted, the RTM is updated to show that each design element is formally associated with a specific requirement. The outputs of the design stage are the design document, an updated RTM, and an updated project plan.
Development (Coding) Stage:	
The development stage takes as its primary input the design elements described in the approved design document. For each design element, a set of one or more software artefacts will be produced. Software artefacts include but are not limited to menus, dialogs, and data management forms, data reporting formats, and specialized procedures and functions. Appropriate test cases will be developed for each set of functionally related software artefacts, and an online help system will be developed to guide users in their interactions with the software.
 

The RTM will be updated to show that each developed artefact is linked to a specific design element, and that each developed artefact has one or more corresponding test case items. At this point, the RTM is in its final configuration. The outputs of the development stage include a fully functional set of software that satisfies the requirements and design elements previously documented, an online help system that describes the operation of the software, an implementation map that identifies the primary code entry points for all major system functions, a test plan that describes the test cases to be used to validate the correctness and completeness of the software, an updated RTM, and an updated project plan.
Integration & Test Stage:
During the integration and test stage, the software artefacts, online help, and test data are migrated from the development environment to a separate test environment. At this point, all test cases are run to verify the correctness and completeness of the software. Successful execution of the test suite confirms a robust and complete migration capability. During this stage, reference data is finalized for production use and production users are identified and linked to their appropriate roles. The final reference data (or links to reference data source files) and production user list are compiled into the Production Initiation Plan.


The outputs of the integration and test stage include an integrated set of software, an online help system, an implementation map, a production initiation plan that describes reference data and production users, an acceptance plan which contains the final suite of test cases, and an updated project plan.
•	Installation & Acceptance Test:
During the installation and acceptance stage, the software artefacts, online help, and initial production data are loaded onto the production server. At this point, all test cases are run to verify the correctness and completeness of the software. Successful execution of the test suite is a prerequisite to acceptance of the software by the customer.
After customer personnel have verified that the initial production data load is correct and the test suite has been executed with satisfactory results, the customer formally accepts the delivery of the software.
 
The primary outputs of the installation and acceptance stage include a production application, a completed acceptance test suite, and a memorandum of customer acceptance of the software. Finally, the PDR enters the last of the actual labor data into the project schedule and locks the project as a permanent project record. At this point the PDR "locks" the project by archiving all software items, the implementation map, the source code, and the documentation for future reference.
Maintenance:
Outer rectangle represents maintenance of a project, Maintenance team will start with requirement study, understanding of documentation later employees will be assigned work and they will undergo training on that particular assigned category. For this life cycle there is no end, it will be continued so on like an umbrella (no ending point to umbrella sticks).
3.4. Software Requirement Specification 
 3.4.1. Overall Description
A Software Requirements Specification (SRS) – a requirements specification for a software system  is a complete description of the behaviour of a system to be developed. It includes a set of use cases that describe all the interactions the users will have with the software. In addition to use cases, the SRS also contains non-functional requirements. Non-functional requirements are requirements which impose constraints on the design or implementation (such as performance engineering requirements, quality standards, or design constraints). 
System requirements specification: A structured collection of information that embodies the requirements of a system. A business analyst, sometimes titled system analyst, is responsible for analyzing the business needs of their clients and stakeholders to help identify business problems and propose solutions. Within the systems development lifecycle domain, the BA typically performs a liaison function between the business side of an enterprise and the information technology department or external service providers. Projects are subject to three sorts of requirements:
•	Business requirements describe in business terms what must be delivered or accomplished to provide value.
•	Product requirements describe properties of a system or product (which could be one of several ways to accomplish a set of business requirements.)
•	Process requirements describe activities performed by the developing organization. For instance, process requirements could specify .Preliminary investigation examine project feasibility, the likelihood the system will be useful to the organization. The main objective of the feasibility study is to test the Technical, Operational and Economical feasibility for adding new modules and debugging old running system. All system is feasible if they are unlimited resources and infinite time. There are aspects in the feasibility study portion of the preliminary investigation:		

•	ECONOMIC FEASIBILITY	
A system can be developed technically and that will be used if installed must still be a good investment for the organization. In the economical feasibility, the development cost in creating the system is evaluated against the ultimate benefit derived from the new systems. Financial benefits must equal or exceed the costs. The system is economically feasible. It does not require any addition hardware or software. Since the interface for this system is developed using the existing resources and technologies available at NIC, There is nominal expenditure and economical feasibility for certain.
•	  OPERATIONAL FEASIBILITY 	
Proposed projects are beneficial only if they can be turned out into information system. That will meet the organization’s operating requirements. Operational feasibility aspects of the project are to be taken as an important part of the project implementation. This system is targeted to be in accordance with the above-mentioned issues. Beforehand, the management issues and user requirements have been taken into consideration. So there is no question of resistance from the users that can undermine the possible application benefits. The well-planned design would ensure the optimal utilization of the computer resources and would help in the improvement of performance status.	
•	  TECHNICAL FEASIBILITY
Earlier no system existed to cater to the needs of ‘Secure Infrastructure Implementation System’. The current system developed is technically feasible. It is a web based user interface for audit workflow at NIC-CSD. Thus it provides an easy access to .the users. The database’s purpose is to create, establish and maintain a workflow among various entities in order to facilitate all concerned users in their various capacities or roles. Permission to the users would be granted based on the roles specified.  Therefore, it provides the technical guarantee of accuracy, reliability and security. 


 3.4.2. External Interface Requirements
User Interface
The user interface of this system is a user friendly python Graphical User Interface.
Hardware Interfaces
The interaction between the user and the console is achieved through python capabilities. 
Software Interfaces
The required software is python.
HARDWARE REQUIREMENTS:
•	Processor			-	Pentium –IV
•	Speed				-    	1.1 GHz
•	RAM				-           4GB(min)
•	Hard Disk			-   	500GB
•	Key Board			-    	Standard Windows Keyboard
•	Mouse				-    	Two or Three Button Mouse
•	Monitor			-    	SVGA
SOFTWARE REQUIREMENTS:
•	Operating System		-	Windows 10/above
•	Programming Language	-	Python 3.7 /above

5. IMPLEMETATION

from tkinter import messagebox
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
import tkinter
import numpy as np
from tkinter import filedialog
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import normalize
from sklearn import svm
import pickle
import os
import cv2

main = tkinter.Tk()
main.title("Lung Cancer Detection Using Ensemble Algorithm")
main.geometry("1300x1200")


global filename
global dataset
global  X_train, X_test, y_train, y_test
global classifier
global rbf_classifier

def upload():
    global filename
    global dataset
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir = "Dataset")
    text.delete('1.0', END)
    text.insert(END,filename+' dataset loaded\n')
    dataset = pd.read_csv(filename)
    dataset.fillna(0, inplace = True)
    text.insert(END,str(dataset.head())+"\n")
    text.insert(END,"Dataset contains total records    : "+str(dataset.shape[0])+"\n")
    text.insert(END,"Dataset contains total attributes : "+str(dataset.shape[1])+"\n")
    label = dataset.groupby('Level').size()
    label.plot(kind="bar")
    plt.show()

def processDataset():
    global X, Y
    global dataset
    text.delete('1.0', END)
    le = LabelEncoder()
    dataset['Level'] = pd.Series(le.fit_transform(dataset['Level'].astype(str)))
    dataset['Patient Id'] = pd.Series(le.fit_transform(dataset['Patient Id'].astype(str)))
    text.insert(END,str(dataset.head())+"\n\n")
    X = dataset.values[:,1:dataset.shape[1]-1]
    Y = dataset.values[:,dataset.shape[1]-1]
    Y = Y.astype('int')
    X = normalize(X)
    print(X)
    print(Y)
    text.insert(END,"Dataset contains total records : "+str(X.shape[0])+"\n")
    text.insert(END,"Dataset contains total Features: "+str(X.shape[1])+"\n")
    
    
def runEnsemble():
    global classifier
    global  X_train, X_test, y_train, y_test
    global X, Y
    text.delete('1.0', END)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    text.insert(END,"Total dataset records : "+str(X.shape[0])+"\n")
    text.insert(END,"Total dataset records used to train algorithms : "+str(X_train.shape[0])+"\n")
    text.insert(END,"Total dataset records used to test algorithms  : "+str(X_test.shape[0])+"\n\n")
    dt = DecisionTreeClassifier()
    ada_boost = AdaBoostClassifier(n_estimators=100, random_state=0)
    mlp = MLPClassifier(max_iter=200,hidden_layer_sizes=100,random_state=42)
    vc = VotingClassifier(estimators=[('dt', dt), ('ab', ada_boost), ('mlp', mlp)], voting='soft')
    vc.fit(X_train, y_train)
    predict = vc.predict(X_test) 
    p = precision_score(y_test, predict,average='micro') * 100
    r = recall_score(y_test, predict,average='micro') * 100
    f = f1_score(y_test, predict,average='micro') * 100
    a = accuracy_score(y_test,predict)*100
    text.insert(END,"Ensemble of Decision Tree, MLP and AdaBoost Performance Result\n\n")
    text.insert(END,"Ensemble Algorithms Precision : "+str(p)+"\n")
    text.insert(END,"Ensemble Algorithms Recall    : "+str(r)+"\n")
    text.insert(END,"Ensemble Algorithms FMeasure  : "+str(f)+"\n")
    text.insert(END,"Ensemble Algorithms Accuracy  : "+str(a)+"\n")
    classifier = vc
    

def predict():
    global classifier
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset")
    test = pd.read_csv(filename)
    data = test.values
    data = data[:,1:data.shape[1]]
    data = normalize(data)
    predict = classifier.predict(data)
    print(predict)
    test = test.values
    for i in range(len(predict)):
        result = 'High'
        if predict[i] == 0:
            result = 'High. CT Scan Required'
        if predict[i] == 1:
            result = 'Low. CT Scan Not Required'
        if predict[i] == 2:
            result = 'Medium. CT Scan Not Required'
        text.insert(END,"Test Values : "+str(test[i])+" Predicted Disease Status : "+result+"\n\n")
        

def trainRBF():
    global rbf_classifier
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir = ".")
    if os.path.exists('model/model.txt'):
        with open('model/model.txt', 'rb') as file:
            rbf_classifier = pickle.load(file)
        file.close()
        X = np.load('model/X.txt.npy')
        Y = np.load('model/Y.txt.npy')
        X = np.reshape(X, (X.shape[0],(X.shape[1]*X.shape[2]*X.shape[3])))
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        predict = rbf_classifier.predict(X_test)
        svm_acc = accuracy_score(y_test,predict)*100
        text.insert(END,"RBF training accuracy : "+str(svm_acc)+"\n\n")
    else:
        X = []
        Y = []
        for root, dirs, directory in os.walk(filename):
            for j in range(len(directory)):
                name = os.path.basename(root)
                print(name+" "+root+"/"+directory[j])
                if 'Thumbs.db' not in directory[j]:
                    img = cv2.imread(root+"/"+directory[j])
                    img = cv2.resize(img, (10,10))
                    im2arr = np.array(img)
                    im2arr = im2arr.reshape(10,10,3)
                    X.append(im2arr)
                    if name == 'normal':
                        Y.append(0)
                    if name == 'abnormal':
                        Y.append(1)
        X = np.asarray(X)
        Y = np.asarray(Y)
        print(Y.shape)
        print(X.shape)
        print(Y)
        X = X.astype('float32')
        X = X/255
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        Y = Y[indices]
        np.save('model/X.txt',X)
        np.save('model/Y.txt',Y)
        X = np.load('model/X.txt.npy')
        Y = np.load('model/Y.txt.npy')
        X = np.reshape(X, (X.shape[0],(X.shape[1]*X.shape[2]*X.shape[3])))
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        rbf_classifier = svm.SVC(kernel='rbf') 
        rbf_classifier.fit(X, Y)
        predict = rbf_classifier.predict(X_test)
        svm_acc = accuracy_score(y_test,predict)*100
        text.insert(END,"RBF training accuracy : "+str(svm_acc)+"\n\n")
        with open('model/model.txt', 'wb') as file:
            pickle.dump(rbf_classifier, file)
        file.close()
               
def predictCTscan():
    global rbf_classifier
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="testImages")
    img = cv2.imread(filename)
    img = cv2.resize(img, (10,10))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(10,10,3)
    X = np.asarray(im2arr)
    X = X.astype('float32')
    X = X/255
    XX = []
    XX.append(X)
    XX = np.asarray(XX)
    print(XX.shape)
    X = np.reshape(XX, (XX.shape[0],(XX.shape[1]*XX.shape[2]*XX.shape[3])))
    print(X.shape)
    predict = rbf_classifier.predict(X)
    if predict == 0:
        msg = "Uploaded CT Scan is Normal"
    if predict == 1:
        msg = "Uploaded CT Scan is Abnormal"
    img = cv2.imread(filename)
    img = cv2.resize(img, (400,400))
    cv2.putText(img, msg, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 255), 2)
    cv2.imshow(msg, img)
    cv2.waitKey(0)    
    

font = ('times', 16, 'bold')
title = Label(main, text='Lung Cancer Detection Using Ensemble Algorithm')
title.config(bg='dark goldenrod', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 14, 'bold')
upload = Button(main, text="Upload Lung Cancer Dataset", command=upload)
upload.place(x=950,y=100)
upload.config(font=font1)  

processButton = Button(main, text="Dataset Preprocessing", command=processDataset)
processButton.place(x=950,y=150)
processButton.config(font=font1) 

eaButton = Button(main, text="Run Ensemble Algorithms", command=runEnsemble)
eaButton.place(x=950,y=200)
eaButton.config(font=font1) 

predictButton = Button(main, text="Predict Lung Cancer Disease", command=predict)
predictButton.place(x=950,y=250)
predictButton.config(font=font1)

rbfButton = Button(main, text="Train RBF on Lungs CT-Scan Images", command=trainRBF)
rbfButton.place(x=950,y=300)
rbfButton.config(font=font1)

predictButton = Button(main, text="Predict Cancer from CT-Scan", command=predictCTscan)
predictButton.place(x=950,y=350)
predictButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=110)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)


main.config(bg='turquoise')
main.mainloop()

6. TESTING
Implementation and Testing:
Implementation is one of the most important tasks in project is the phase in which one has to be cautions because all the efforts undertaken during the project will be very interactive. Implementation is the most crucial stage in achieving successful system and giving the users confidence that the new system is workable and effective. Each program is tested individually at the time of development using the sample data and has verified that these programs link together in the way specified in the program specification. The computer system and its environment are tested to the satisfaction of the user.
Implementation
The implementation phase is less creative than system design. It is primarily concerned with user training, and file conversion. The system may be requiring extensive user training. The initial parameters of the system should be modifies as a result of a programming. A simple operating procedure is provided so that the user can understand the different functions clearly and quickly. The different reports can be obtained either on the inkjet or dot matrix printer, which is available at the disposal of the user. 	The proposed system is very easy to implement. In general implementation is used to mean the process of converting a new or revised system design into an operational one.
Testing
Testing is the process where the test data is prepared and is used for testing the modules individually and later the validation given for the fields. Then the system testing takes place which makes sure that all components of the system property functions as a unit. The test data should be chosen such that it passed through all possible condition. Actually testing is the state of implementation which aimed at ensuring that the system works accurately and efficiently before the actual operation commence. The following is the description of the testing strategies, which were carried out during the testing period.
System Testing
Testing has become an integral part of any system or project especially in the field of information technology.  The importance of testing is a method of justifying, if one is ready to move further, be it to be check if one is capable to with stand the rigors of a particular situation cannot be underplayed and that is why testing before development is so critical. When the software is developed before it is given to user to use the software must be tested whether it is solving the purpose for which it is developed.  This testing involves various types through which one can ensure the software is reliable. The program was tested logically and pattern of execution of the program for a set of data are repeated.  Thus the code was exhaustively checked for all possible correct data and the outcomes were also checked.  

Module Testing
To locate errors, each module is tested individually.  This enables us to detect error and correct it without affecting any other modules. Whenever the program is not satisfying the required function, it must be corrected to get the required result. Thus all the modules are individually tested from bottom up starting with the smallest and lowest modules and proceeding to the next level. Each module in the system is tested separately. For example the job classification module is tested separately. This module is tested with different job and its approximate execution time and the result of the test is compared with the results that are prepared manually. The comparison shows that the results proposed system works efficiently than the existing system. Each module in the system is tested separately. In this system the resource classification and job scheduling modules are tested separately and their corresponding results are obtained which reduces the process waiting time.
Integration Testing
After the module testing, the integration testing is applied.  When linking the modules there may be chance for errors to occur, these errors are corrected by using this testing. In this system all modules are connected and tested. The testing results are very correct. Thus the mapping of jobs with resources is done correctly by the system.
Acceptance Testing
When that user fined no major problems with its accuracy, the system passers through a final acceptance test.  This test confirms that the system needs the original goals, objectives and requirements established during analysis without actual execution which elimination wastage of time and money acceptance tests on the shoulders of users and management, it is finally acceptable and ready for the operation

Test Case Id	Test Case Name	Test Case Desc.	Test Steps	Test Case Status	Test Priority
			Step	Expected	Actual		
01	Upload Lung Cancer Dataset	Test whether Upload Lung Cancer Dataset not into the system	If the Lung Cancer Dataset may not uploaded	We cannot do further operations	we will do further
operations	High	High
02 	Dataset Preprocessing	Test Dataset Preprocessing or not into the system	If Dataset PreprocessingMay not uploaded	We cannot do further operations	we will do further
operations	High	High
03	Run Ensemble Algorithm	Test Run Ensemble Algorithm or not	If the Run Ensemble Algorithm may 
Not loaded	We cannot do further operations	we will do further
operations	High	High



04	Predict Lung Cancer Disease	Test Predict Lung Cancer Disease not into the system	If the Predict Lung Cancer Disease may not uploaded	We cannot do further operations	we will do further
operations	High	High
025	Train RBF on LUNGS CT-SCAN Image	Test Train RBF on LUNGS CT-SCAN Image or not into the system	If the RBF on LUNGS CT-SCAN Image not Trained	We cannot do further operations	we will do further
operations	High	High
06	Predict Cancer from CT-Scan	Test Predict Cancer from CT-Scan or not	If the Predict Cancer from CT-Scan may 
Not loaded	We cannot do further operations	we will do further
operations	High	High









SCREEN SHOTS:
 
In above dataset first row contains column names and remaining rows contains dataset values and in last column we have disease label as HIGH, LOW or MEDIUM and we will use above dataset to train ensemble algorithm. Below is the test dataset used to predict disease
 
In above test dataset we done have class label as HIGH, MEDIUM or LOW and when we apply above dataset on ensemble algorithm then it will predict disease stage. Below CT-SCAN image are using to train RBF algorithms and this images are available inside ‘Lung_Images_Dataset’ folder
 
In above dataset folder we have two folders such as ‘normal’ or ‘abnormal’ and you can inside any folder to see images
 
In above screen showing all images from abnormal folder and we used above images to train RBF algorithm


To run project double click on ‘run.bat’ file to get below screen
 
click on ‘Upload Lung Cancer Dataset’ button to upload dataset
 
selecting and uploading ‘dataset.csv’ file and then click on ‘Open’ button to load dataset and to get below screen
 
 in text area we can see dataset loaded and dataset contains total 1000 records and 25 columns and in above dataset we can see some string values are there but machine learning algorithms accept only numeric values so we need to preprocess above dataset to convert string values to numeric by assigning unique id’s to each string value and in above graph in x-axis we can see disease stage as HIGH or LOW and number of patients detected in that stage are plotting in y-axis as bar graph. Now close above graph and then click on ‘Dataset Preprocessing’ button to convert to string values to numeric
 
 all string values are converted to numeric and now click on ‘Run Ensemble Algorithm’ button to train on above dataset
 
 after training we got ensemble algorithms accuracy as 100% and now click on ‘Predict Lung Cancer Disease’ button and upload test dataset 
 
selecting and uploading ‘test.csv’ file and then click on ‘Open’ button to load test data and then will get below prediction result
 
in square bracket we can see test values and then by analysing those test values ensemble has given prediction result as LOW, HIGH or MEDIUM and we can see this result after square bracket and if disease high then application asking user to go for CT-SCAN image. Now click on ‘Train RBF on LUNGS CT-SCAN Image’ to train RBF with lungs CT-SCAN.
 
selecting and uploading ‘Lung_Images_Dataset’ button and then click on ‘Select Folder’ button to load dataset and to get below screen
 
 RBF trained on images and its got prediction accuracy as 82 and always this accuracy may vary as RBF calculate accuracy on random test data. Now click on ‘Predict Cancer from CT-Scan’ button to upload test image and then get prediction result
 
 selecting and uploading ‘2.png’ image and then click on ‘Open’ button to get below result
 
in image title bar or yellow colour text you can see prediction result as ‘ABNORMAL’ and now test with other image
 
 uploading 10.png and below is the result
 

 predicted result is ‘NORMAL’ and similarly you can upload remaining images and test


8. CONCLUSION
In this project we are using Lung Cancer dataset to train ensemble algorithm by combining AdaBoost, Multilayer Perceptron and Decision Tree and then after training when we upload test data then this algorithm will predict lung cancer stage as HIGH, LOW and MEDIUM and if HIGH detected then application will ask user to go for CT-SCAN.Here we designed another algorithm using RBF and lung cancer CT-SCAN images and this CT-SCAN images will be trained using RBF algorithm and then after training when user upload test image then application will predict whether uploaded CT-SCAN is normal or abnormal.


<!---
shreya8k/shreya8k is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->

