# IoT-anomaly-detector (IoT-AD)

This GitHub repository contains a collection of scripts for analyzing device actions in a smart home environment. The scripts are designed to preprocess data, train models, detect anomalies, and rollback device actions if necessary. The following is an overview of the scripts included in this project: 

**Scripts:** 

**1. DeviceStateData.py:** This script controls the device actions in a smart home by monitoring and managing the state of devices.  

**2. Preprocess.py:** This script converts pcap files to CSV format, making it easier to analyze the data and extract meaningful information.  

**3. DeviceProfileTrain.py:** In this script, the machine learning model is trained using device action data, and a signature is created for each device's action pattern.  

**4. trainandtest.py:** This script focuses on training the machine learning model using multiple device actions and their interactions. It also includes testing and evaluation of the trained model. 

**5. complexmodel.py:** This script builds a deep learning model that can handle complex interactions and multiple device actions. It trains the model using appropriate data and fine-tunes it for optimal performance.  

**6. validation.py:** Using the trained model, this script performs anomaly detection on device actions. It identifies any unusual or unexpected behavior and flags them as anomalies.  

**7. Rollback.py:** In the event of detecting interaction anomalies, this script rolls back the device actions to their stable state, ensuring the integrity and security of the smart home environment. 

**Getting Started:**  

To use these scripts, follow these steps:  

1. Clone the repository to your local machine.  

2. Install the necessary dependencies and libraries required for running the scripts.  

3. Run each script in the specified sequence as described above, ensuring the proper data inputs and configuration settings. 
