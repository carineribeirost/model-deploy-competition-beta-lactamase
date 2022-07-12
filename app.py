import streamlit as st
import pandas as pd
from PIL import Image
import os
import base64
import pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import mols2grid
import streamlit.components.v1 as components



def prob(input_data):
    """
     This function receives a dataset and calculate the
     probabilities of activity of different molecules 
     in a pre defined model
    """
    #load model
    load_model = pickle.load(open('model.pkl', 'rb'))
    # Apply model to make predictions
    prediction = load_model.predict_proba(input_data)
    prob_list = list()
    for i in prediction:
        prob_list.append(round(i[1] * 100, 2))
    return prob_list
    
def near_five(number):
    """
    Round down a number to its nearest multiple of five
    """
    return 5 * int (number/5)

#Molecular descriptor calculator
def _compute_single_fp_descriptor(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception as E:
        return None

    if mol:
        fp = Chem.AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        return np.array(fp)
    
    return None
    
#Computes ecfp descriptors  
def compute_fp_descriptors(smiles_list):
    
    idx_to_keep = list()
    descriptors = list()
    for i, smiles in enumerate(smiles_list):
        fp = _compute_single_fp_descriptor(smiles)
        if fp is not None:
            idx_to_keep.append(i)
            descriptors.append(fp)

    return np.vstack(descriptors), idx_to_keep
@st.cache
def desc_calc(df):
    """
    Calculate Morgan Fingerprints with radius 2 for a list of smiles
    """
    smiles_array = np.array(df)
    descriptors, idx = compute_fp_descriptors(smiles_array)
    df_d = pd.DataFrame(descriptors)
    df_d.to_csv('descriptors_output.csv')

# File download
def filedownload(df):
    """
    Make the dataset with prediction probabilities downloadable
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download Predictions</a>'
    return href

# Model building
def build_model(input_data):
    """
    returns activity probilities and concatenates it with molecule smiles and molecule id
    """
    probabilities = prob(input_data)
    st.header('**Prediction output**')
    prediction_output = pd.Series(probabilities, name='Bioactivity Prediction (%)')
    molecule_name = pd.Series(load_data[1], name='molecule_name')
    df = pd.concat([molecule_name, prediction_output], axis=1)
    return df

def slider_mol(min_value = 10, max_value = 90):
    """
    instantiate an slider for streamlit with given parameters
    """
    prob_cutoff = st.slider(
        label="Show compounds with that probability below:",
        min_value=min_value,
        max_value=max_value,
        value=50,
        step=1)
    return prob_cutoff
    
    
    
#Logo image
#image = Image.open('logo.png')

#st.image(image, use_column_width=True)



# Page title
st.markdown("""
# Beta-Latacmase Bioctivity Predictor App 
This app allows you to predict the bioactivity towards inhibting the `Beta-lactamase` enzyme. `Beta-lactamase` is a target for antibiotic resistance.
- App built in `Python` + `Streamlit` by [Carine Ribeiro]
- Descriptor calculated using RDkit Morgan FIngerprint with radius 2.
---
""")

# Sidebar
#it will access files for prediction
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input file", type=['txt'])
    st.sidebar.markdown("""
[Example input file](https://raw.githubusercontent.com/dataprofessor/bioactivity-prediction-app/main/example_acetylcholinesterase.txt)
""")
    


if st.sidebar.checkbox('Check to Predict'): #a checkbox to run the prediction
    load_data = pd.read_table(uploaded_file, sep='\t', header=None) #read received molecule data
    smiles_list = list()
    #put smiles in a list
    for i in load_data[0]:
        smiles_list.append(i)
    
    #show input data
    st.header('**Original input data**')
    st.write(load_data)

    #calculate descriptors
    with st.spinner("Calculating descriptors..."):
        desc_calc(smiles_list)

    # Read in calculated descriptors and display the dataframe
    desc = pd.read_csv('descriptors_output.csv')
    
    # Read descriptor list used in previously built model
    Xlist = list(pd.read_csv('molecule_descriptors_lvhc.csv', index_col = [0]).columns)
    desc_subset = desc[Xlist]
    # Apply trained model to make prediction on query compounds
    df = build_model(desc_subset)
    df_completed = pd.concat([pd.Series(smiles_list, name = 'SMILES'), df], axis = 1)    
    
    #edit dataframe to show only values with probability greater than cutoff
    prob_cutoff = slider_mol(min_value = near_five(min(df['Bioactivity Prediction (%)'])),max_value = near_five(max(df['Bioactivity Prediction (%)'])))
    df_result = df_completed[df_completed["Bioactivity Prediction (%)"] > prob_cutoff]

    st.write(df_result)
    st.markdown(filedownload(df_result), unsafe_allow_html=True)
    
    raw_html = mols2grid.display(df_result, mapping={"smiles": "SMILES"})._repr_html_()
    components.html(raw_html, width=900, height=900, scrolling=True)
