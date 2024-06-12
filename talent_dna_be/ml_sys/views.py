from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import tensorflow as tf

# Load the assessment model
model_path = 'ml_sys/model/multi_regress_assess.h5'
model = tf.keras.models.load_model(model_path)
talents = ['CR', 'GN', 'CP', 'NB', 'AC', 'RS', 'ST', 'DR', 'EX', 'FL', 'PS', 'SZ',
           'GG', 'FX', 'IT', 'EN', 'SC', 'IN', 'CO', 'AV', 'FC', 'DV', 'CT', 'LG',
           'AR', 'CX', 'CL', 'SF', 'IV', 'EQ', 'VS', 'SG', 'OP', 'HM', 'CV', 'VG',
           'AF', 'AD', 'GS', 'TS', 'CRG', 'DC', 'PF', 'FG', 'AU']


@api_view(['GET'])
def simple_api_view(request):
    data = {"message": "Hello, TalentDNA"}
    return Response(data)


@api_view(['POST'])
def ml_process(request):

    # <--- Start: Talents Multi Regression Process --->
    # Define input into list
    input = prop_input(request)
    input = np.array(input).astype(float)

    # Standarization
    X = standar_scaler(input)

    # Predicting talents
    inference_result = model.predict([X])

    # Construct dataframe for output
    talents_result_sorted = construct_talents_df(inference_result)

    # Return the result as JSON
    return Response(talents_result_sorted, status=status.HTTP_200_OK)


def prop_input(input):
    input_string = input.data.get('string', '')
    processed_data = [[int(char) for char in input_string]]
    return processed_data


def standar_scaler(input):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(input.reshape(-1, 1)).flatten().tolist()
    return scaled_data


def construct_talents_df(input):
    flattened_inference_res = np.ravel(input)

    result_df = pd.DataFrame({
        'Talent': talents,
        'Predicted Rank': flattened_inference_res,
    })

    result_df_sorted = result_df.sort_values(
        by='Predicted Rank', ascending=False)
    result_df_sorted.reset_index(drop=True, inplace=True)

    return result_df_sorted
