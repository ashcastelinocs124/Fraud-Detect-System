from transformers import AutoProcessor, AutoModelForVisionText2Text
from PIL import Image
import faiss
import numpy as np
import re
import logging
import os
from langchain_ollama import OllamaEmbeddings
from django.db import models
from difflib import SequenceMatcher
from fuzzywuzzy import fuzz
from tensorflow.keras.applications import resnet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import sklearn as sk
import netoworkx from nx
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_spiit,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, roc_auc_score, classification_report, confusion_matrix, roc_curve



#metrics for analyzing data
HISTORICAL_DATA_DIM = 768  # Nomic embedding dimension
VENDOR_DATA_DIM = 768  # Nomic embedding dimension
UNUSUAL_CHARS_THRESHOLD = 5
INCONSISTENT_CAPS_THRESHOLD = 0.3
VENDOR_SIMILARITY_THRESHOLD = 0.7
HISTORICAL_SIMILARITY_THRESHOLD = 0.6
HISTORICAL_PAYMENT_THRESHOLD = 0.4
HIGH_VALUE_THRESHOLD = 10000

#Data Macthing Logic
name_similarity_model = 0.40
description_similarity_model = 0.30
image_similarity = 0.20
price_similarity_model = 0.10
threeshold_for_match = 0.80

#Knowledhge graph is a network representation of entities and their relationships. It represents a real world data in a way that highlights the relationships and dependencies between different pieces of information.

#Represents semantic relationships between 2 variables
#Nodes connected by edges



#In this case 
#Nodes - Accounts, Transactions, Products
#Edges - Relationships between them

#Utilizes natural language processing


model = resnet50.ResNet50(weights = "imagenet", include_top = False, pooling = "avg")



class Knowledgegraph:
    G = nx.Graph()
    entities = [
        ("Account1", "Transaction1"),
        ("Account2", "Transaction1"),
        ("Account1","Device1"),
        ("Account2","Device2")
    ]
    G.add_edges_from(entities)

    plt.figure(figsize = (10,8))
    pos = 




class algorithim:
    X = data.drop(["Class"], axis=1)
    y = data.Class
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    list_of_split_data = [X_train_resampled, X_test, y_train_resampled, y_test
                              
    def __init__(self, splitdata):
        self.list_of_split_data = list_of_split_data
        

    def randomforestalgorithim(data):
        
        clf = RandomForestClassifier()
        clf.fit(slef.list_of_split_data[0], slef.list_of_split_data[2])
        
        pred = clf.predict(self.list_of_split[1])
        print(accuracy_score(self.list_split_data[3], pred) * 100
              
    def CNN(data):
        scaler = StandardScaler() #Standardizes the features by removing the mean and scaling to unit variance
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        y_train = y_train.to_numpy()
        y_test = y_test.to_numpy()

        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],1)


        #Build the CNN Model
        
        
    





class databasemodel:

    class Purchase(models.Model):
        description = models.TextField()
        total_amount = models.DecimalField(max_digits = 10, decimal_places = 2)

     Transaction(models.Model):
        purchase = models.ForeignKey(Purchase, on_delete = models.CASCADE, related_name = "transactions")
        amount = models.DecimalField(max_digits = 10, decimal_places =2 )
        transaction_date = models.DateTimeField(auto_now_add = True)

    def split_purchase(description, total_amount, split_amounts):
        purchase = Purchase.objects.create(description = description, total_amount = total_amount)

        for amount in split_amounts:
            Transaction.objects.create(purchase = purchase, amount = amount)
        return purchase
class purchasefraud:#Helps figure out the purchase
    def itemfraud(item1, item2):
        name_similarity = fuzz.token_sort_ratio(item1['name'], item2['name'])
        desc_similarity = fuzz.token_sort_ratio(item1['name'], item2['name'])

        price_similarity= 1 - abs(item1['price'] - item2['price']) / item2['price']

        total_score = (name_similarity_model * name_similarity) + (description_similarity_model * desc_similarity) + (price_similarity_model * price_similarity)
        return total_score
    def extractfeatures(img_path):
        img = image.load_img(img_path, target_size = (224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis = 0)
        img_data = preprocess_input(img_data)
        features = model.predict(img_data)

        return features
    def main():
        fakeimage = input("Enter path for fakeimage")
        realimage = input("Enter path for realimage")
        fake_features = extractfeatures(fakeimage)
        real_features = extractfeatures(realimage)

        cosine_similarity = np.dot(fake_features, real_featuers.T) / (np.linalg.norm(fake_features) * np.lingalg.norm(real_features))

        if (cosine_similarity[0][0] > 0.9):
            print("Images are similar")
        else:
            print("Images are different")


        
class sellerfraud:
    def algorithim():
        








class InvoiceFraudDetectionSystem:
    def __init(self, historical_index_path, vendor_index_path):
        self.model = AutoModelForVisionText2Text.from_pretrained("microsoft/Phi-3-vision-128k-instruct")
        self.processor = AutoProcessor.from_pretrained("microsoft/Phi-3-vision-128k-instruct")
        self.historical_data = self.load_index(historical_index_path,HISTORICAL_DATA_DIM)
        self.vendor_details = self.load_index(vendor_index_path,VENDOR_DATA_DIM)
        self.embeddings = OllamaEmbeddings(model = "nomic-embed-text")
        
    def detect_altered_text(self, analysis):
        unusual_chars = re.findall(r'[^\w\x,.$]', analysis)
        if(len(unusual_chars) > UNUSUAL_CHARS_THRESHOLD):
            return true
        word = analysis.split()
        inconsistent_caps = sum(1 for word in words if word.istitle() != words[0].istitle())
        if (inconsistent_caps > len(words) * INCONSISTENT_CAPS_THRESHOLD):
            return True
        return False
    def get_embedding(self, text):
        return self.embeddings.embed_query(text)
    def detect_multiple_payments(self, paymentcount):
        if (paymentcount > HISTORICAL_PAYMENT_THRESHOLD):
            return true
        return false

    def detect_inconsistent_formatting(self, analysis):
        data_formats = re.findall(r'\d{1,4}[-/]d{1,2}[-/]d{1,4}', analysis)
        if len(set(date_formats)) > 1:
            return true
        lines = analysis.split('\n')
        left_aligned  =
    def analyze_invoice(self, image_path):
        try:
            image = image.open(image_path)
        except IOERROR:
            logging.error(f"Unable to open image file: {image_path}")
            return None, ["Error: Unable to process invoice image"]
        inputs = self.model(images = image, return_tensors = "pt")
        outputs = self.model.generate(**inputs, max_new_tokens = 256)
        analysis = self.processor.decode(outputs[0], skip_special_tokens = True)
        anomalies = self.check_anomalies(analysis)
        return analysis, anomalies
    def detect_suspicious_vendor(self, analysis):
        vector = self.get_embedding(analysis)
        D, I = self.vendor_details['index'].search(np.array([vector]),k = 1)
        if D[0][0] > VENDOR_SIMILARITY_THRESHOLD:
            return True
        suspicious_keywords = ['urgent','confidential','immediate payment','offshore']
        if any(keyword in analysis.lower(
    def check_anomalies(self, analysis):
        anomalies = []
        if self.detect_altered_text(analysis):
            anomalies.append("Possible altered text detected")
        if self.detect_inconsistent_formatting(analysis):
            anomalies.append("Incorrect formatting")
        if self.detect_multiple_payments(analysis):
            anomalies.append("Too many payments")
