import torch
import streamlit as st
import seaborn as sns
import imgaug as ia
import pandas as pd
sns.set_theme(style="darkgrid")
sns.set()
st.title('ITR EXTRACTOR')
from test_ocr import return_text
import os 
import cv2

class ITRExtractor:
    def __init__(self, image_path = None):
        #load ultralytics yolov5 and assign best_weights.pt as the file containing best weights 
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='best_weights.pt')
        self.image_path = image_path
        #15 classes
        self.classes = [ 'ITR type','Name of Customer','Town name','State Name','Assessment Year','PAN Number',
                        'Gross Salary','Deduction','Net Salary','Current year loss','total tax paid','Exempt income',
                        'Acknowledgement Number','Date','Status']
        self.image = cv2.imread(self.image_path)

    def main(self):
        dictionary = {}
        boxes = []
        results = self.model(self.image_path, size = 1920)
        #convert detection results into a dataframe
        df = results.pandas().xyxy[0]
        for i,cls in enumerate(self.classes):
            #for multiple detections for a class, select the one with max confidence
            subset = df['class'] == i
            x = []
            for i,flag in enumerate(subset):
                if flag == True:
                    x.append(i)
            #if no detections are there for a class move to the next class
            if len(x)==0:
                dictionary[cls] = ''
                continue
            max_conf_ind = x[0]
            max_conf = 0.0
            for ind in x:
                if df.at[ind,'confidence']>max_conf:
                    max_conf = df.at[ind,'confidence']
                    max_conf_ind = ind

            x1 = df.at[max_conf_ind,'xmin']
            y1 = df.at[max_conf_ind,'ymin']
            x2 = df.at[max_conf_ind,'xmax']
            y2 = df.at[max_conf_ind,'ymax']
            confidence = df.at[max_conf_ind,'confidence']
            label = cls 
            #store bounding box
            boxes.append([x1,y1,x2,y2,confidence,label])     
            #return text from OCR      
            text = return_text(image= self.image, x1= x1, y1=y1, x2=x2, y2=y2)
            #split text
            word_list =  text.split()
            #if OCR fails to return text
            if len(word_list)==0:
                field = ''
            else:
                #select last split
                field = word_list[-1]
            #keep hyphen(for the purpose of date and assessment year) and alphanumeric characters
            field = ''.join(e for e in field if e.isalnum() or e=='-')
            #for numeric classes if field is non-numeric, make it blank
            if cls in ['Gross Salary','Deduction','Net Salary','Current year loss','total tax paid','Exempt income']:
                flag = field.isnumeric()
                if not(flag):
                    field =''
            #store field-value as key-value in a dictionary
            dictionary[cls] = field
        #create object of bounding boxes
        bboxes = ia.BoundingBoxesOnImage([
        ia.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
        for x1, y1, x2, y2, conf, label in boxes], shape = (self.image).shape)
        #draw boxes on image and return
        image_with_boxes = bboxes.draw_on_image(self.image, color=[0, 0, 255], size=2)
        return dictionary, image_with_boxes



if __name__ == '__main__':


    def save_uploaded_file(uploaded_file):

        try:

            with open(os.path.join('images',uploaded_file.name),'wb') as f:

                f.write(uploaded_file.getbuffer())

            return 1    

        except:

            return 0

    uploaded_file = st.file_uploader("Upload Image")

    #if there is a file uploaded
    if uploaded_file is not None:
        
        #if file file has been saved in 'images' folder
        if save_uploaded_file(uploaded_file):

            obj = ITRExtractor(image_path = os.path.join('images',uploaded_file.name))
            #return dictionary and image with detections
            dictionary, image = obj.main()
            #remove image file since work is complete
            os.remove('images/'+uploaded_file.name)
            #display image
            st.image(image)
            st.text('Extracted fields :-')
            #convert dictionary to dataframe
            dframe = pd.DataFrame.from_dict(dictionary.items())
            dframe.columns = ['FIELDS','VALUES']
            #display dataframe as a table
            st.table(dframe)


        


