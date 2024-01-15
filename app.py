# Importing the necessary libraries
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd


app = Flask(__name__)

# Loading the model
model = pickle.load(open('Diseases_classify.pkl', 'rb'))

# Route to render the HTML form
@app.route('/')
def index():
    return render_template('home.html')

# Route to handle the form submission and return the prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get selected symptoms from the form
        selected_symptoms = request.form.get('selectedSymptomsArray')

        # Passing the symptoms to preprocess function
        input_data = preprocess_input(selected_symptoms)
        
        # returning the predicted diseases along with its accuracy to html file as json
        return jsonify({'predicted_disease': input_data})



# Function to preprocess input data before prediction
def preprocess_input(symptom):
    values=[0]*131

    # listing all the symptoms
    symptoms=['itching',' skin_rash',' nodal_skin_eruptions',' dischromic _patches',' continuous_sneezing',
 ' shivering',' chills',' watering_from_eyes',' stomach_pain',' acidity',' ulcers_on_tongue',' vomiting',
 ' cough',' chest_pain',' yellowish_skin',' nausea',' loss_of_appetite',' abdominal_pain',' yellowing_of_eyes',
 ' burning_micturition',' spotting_ urination',' passage_of_gases',' internal_itching',' indigestion',
 ' muscle_wasting',' patches_in_throat',' high_fever',' extra_marital_contacts',' fatigue',' weight_loss',
 ' restlessness',' lethargy',' irregular_sugar_level',' blurred_and_distorted_vision',' obesity',
 ' excessive_hunger',' increased_appetite',' polyuria',' sunken_eyes',' dehydration',' diarrhoea',
 ' breathlessness',' family_history',' mucoid_sputum',' headache',' dizziness',' loss_of_balance',
 ' lack_of_concentration',' stiff_neck',' depression',' irritability',' visual_disturbances',' back_pain',
 ' weakness_in_limbs',' neck_pain',' weakness_of_one_body_side',' altered_sensorium',' dark_urine',
 ' sweating',' muscle_pain',' mild_fever',' swelled_lymph_nodes',' malaise',' red_spots_over_body',
 ' joint_pain',' pain_behind_the_eyes',' constipation',' toxic_look_(typhos)',' belly_pain',' yellow_urine',
 ' receiving_blood_transfusion',' receiving_unsterile_injections',' coma',' stomach_bleeding',
 ' acute_liver_failure',' swelling_of_stomach',' distention_of_abdomen',' history_of_alcohol_consumption',
 ' fluid_overload',' phlegm',' blood_in_sputum',' throat_irritation',' redness_of_eyes',' sinus_pressure',
 ' runny_nose',' congestion',' loss_of_smell',' fast_heart_rate',' rusty_sputum',' pain_during_bowel_movements',
 ' pain_in_anal_region',' bloody_stool',' irritation_in_anus',' cramps',' bruising',' swollen_legs',
 ' swollen_blood_vessels',' prominent_veins_on_calf',' weight_gain',' cold_hands_and_feets',' mood_swings',
 ' puffy_face_and_eyes',' enlarged_thyroid',' brittle_nails',' swollen_extremeties',' abnormal_menstruation',
 ' muscle_weakness',' anxiety',' slurred_speech',' palpitations',' drying_and_tingling_lips',
 ' knee_pain',' hip_joint_pain',' swelling_joints',' painful_walking',' movement_stiffness',
 ' spinning_movements',' unsteadiness',' pus_filled_pimples',' blackheads',' scurring',' bladder_discomfort',
 ' foul_smell_of urine',' continuous_feel_of_urine',' skin_peeling',' silver_like_dusting',' small_dents_in_nails',
 ' inflammatory_nails',' blister',' red_sore_around_nose',' yellow_crust_ooze']
    
    #splitting each symptoms from a string
    symptoms_list = symptom.split(',')
    for x in symptoms_list:
        values[symptoms.index(x)]=1
    values=[values]

    # Create a DataFrame
    df = pd.DataFrame(values, columns=symptoms)

    diseases=['Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis',
       'Drug Reaction', 'Peptic ulcer diseae', 'AIDS', 'Diabetes ',
       'Gastroenteritis', 'Bronchial Asthma', 'Hypertension ', 'Migraine',
       'Cervical spondylosis', 'Paralysis (brain hemorrhage)', 'Jaundice',
       'Malaria', 'Chicken pox', 'Dengue', 'Typhoid', 'hepatitis A',
       'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E',
       'Alcoholic hepatitis', 'Tuberculosis', 'Common Cold', 'Pneumonia',
       'Dimorphic hemmorhoids(piles)', 'Heart attack', 'Varicose veins',
       'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia',
       'Osteoarthristis', 'Arthritis',
       '(vertigo) Paroymsal  Positional Vertigo', 'Acne',
       'Urinary tract infection', 'Psoriasis', 'Impetigo']

    # predicting the diseases based on symptoms passed
    output=model.predict(df)

    return diseases[output[0]-1]

if __name__ == '__main__':
    app.run(debug=True)
