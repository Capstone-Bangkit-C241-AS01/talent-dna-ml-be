from django.conf import settings
from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.calibration import LabelEncoder
import pandas as pd
import numpy as np
import tensorflow as tf
import openai
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

openai.api_key = settings.OPENAI_API_KEY

# Load the assessment model
model_assessment_path = 'ml_sys/model/multi_regress_assess.h5'
model_assessment = tf.keras.models.load_model(model_assessment_path)

# Load the jobs recommenders model
model_jobs_path = 'ml_sys/model/transfer_learning_jobs.h5'
model_jobs = tf.keras.models.load_model(model_jobs_path)

# OpenAI API
llm = ChatOpenAI(api_key=openai.api_key)

# Talents name
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
    inference_result = model_assessment.predict([X])

    # Construct dataframe for output
    talents_result_sorted = construct_talents_df(inference_result)

    # Split talents
    top_10_res, btm_5_res = define_top_and_bottom(talents_result_sorted)

    # Split talents
    top_10_res_1, btm_5_res_1 = output_talent(top_10_res, btm_5_res)
    # <--- End: Talents Multi Regression Process --->

    # <--- Start: Job Recommendation --->
    # Create dataframe for talent and interest
    df_talent_interest = talent_interest(top_10_res)

    # Group by interest
    df_interest_grouped = group_by_intereset(df_talent_interest)

    # Standardized Value
    df_interest_grouped['Standardized Rank'] = standardize_predicted_ranks(
        df_interest_grouped)
    df_interest_grouped = df_interest_grouped.drop(columns="Predicted Rank")

    # Preprocessing
    input_1 = interest_preprop(df_interest_grouped)
    input_1 = np.array(input_1).astype(float)

    # Predicting jobs
    jobs_result = model_jobs.predict(input_1)

    # Fit LabelEncoder with job labels
    df_jobs_interest = pd.read_csv("ml_sys/data/job_interest.csv")
    y_jobs = df_jobs_interest['job_en']
    label_encoder = LabelEncoder()
    label_encoder.fit(y_jobs)

    # Preparing jobs result
    jobs_result = predicted_job(jobs_result, label_encoder, df_jobs_interest)

    # Generating task and work style
    tasks, work_styles = get_job_info_for_predictions(jobs_result)

    # Generating output
    jobs_recommendation = output_for_job_recommend(
        jobs_result, tasks, work_styles)
    # <--- End: Job Recommendation --->

    # <--- Start: Talent Summarization --->
    # Combine descriptions
    combine_top, combine_btm = combine_desc(top_10_res, btm_5_res)

    # Genarate summarization
    top_talent_description, bottom_talent_description = get_summarization(
        combine_top, combine_btm)

    # Response
    response_data = {
        'top_10_talents': top_10_res_1.to_dict(orient='records'),
        'bottom_5_talents': btm_5_res_1.to_dict(orient='records'),
        'top_talent_description': top_talent_description,
        'bottom_talent_description': bottom_talent_description,
        'job_recommendations': jobs_recommendation.to_dict(orient='records'),
    }

    # Return the result as JSON
    return Response(response_data, status=status.HTTP_200_OK)


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


def define_top_and_bottom(input):
    scaler = MinMaxScaler()
    input_reshape = input['Predicted Rank'].values.reshape(-1, 1)
    input['Predicted Rank'] = scaler.fit_transform(input_reshape)

    # Top 10 Talents
    df_top_10 = input.iloc[0:10]

    # Processing strength
    lower_bound_pt = input['Predicted Rank'][9]
    lower_base = lower_bound_pt * (30/45)
    df_top_10['Strength'] = 0
    for index, row in df_top_10.iterrows():
        diff = row['Predicted Rank'] - lower_bound_pt
        df_top_10.at[index, 'Strength'] = lower_base + diff

    # Bottom 5 Talents
    df_btm_5 = input.iloc[40:]

    # Processing strength
    df_btm_5['Strength'] = df_btm_5['Predicted Rank']

    return df_top_10, df_btm_5


def talent_interest(input):
    df_talent = pd.read_csv("ml_sys/data/talents_desc(2).csv")
    df_talent_result = pd.merge(
        input, df_talent, left_on='Talent', right_on='shorten_label')
    df_talent_result = df_talent_result[[
        'Talent', 'Predicted Rank', 'interest']]

    return df_talent_result


def ensure_all_interests(df, required_interests):
    missing_interests = [
        interest for interest in required_interests if interest not in df.index]

    if missing_interests:
        missing_df = pd.DataFrame(
            {'Predicted Rank': [0]*len(missing_interests)}, index=missing_interests)
        df = pd.concat([df, missing_df])

    df = df.reindex(required_interests)

    return df


def group_by_intereset(input):
    interest_grouped = input.groupby('interest').agg({'Predicted Rank': 'sum'})
    required_interests = ['conventional', 'investigative',
                          'enterprising', 'realistic', 'artistic', 'social']
    interest = ensure_all_interests(
        interest_grouped, required_interests=required_interests)
    return interest


def standardize_predicted_ranks(df):
    # Calculate the difference between the highest predicted rank and the second highest
    sorted_ranks = sorted(df['Predicted Rank'], reverse=True)
    max_diff = sorted_ranks[0] - sorted_ranks[1]

    # Initialize standardized values dictionary
    standardized_values = {}

    # Determine if there is a significant difference
    significant_difference = max_diff >= 0.1  # Adjust this threshold as needed

    # Determine the maximum value to be allocated
    max_value = 100 if significant_difference else min(
        90, round(sorted_ranks[0] / sorted_ranks[1] * 90, 2))

    # Identify the category with the highest predicted rank
    max_category = df[df['Predicted Rank'] == sorted_ranks[0]].index[0]

    # Loop through predicted ranks
    for category, rank in df['Predicted Rank'].items():
        if rank == 0:
            standardized_values[category] = 0
        else:
            if category == max_category:
                standardized_values[category] = max_value
            else:
                standardized_values[category] = min(
                    max_value, round(rank / sorted_ranks[0] * max_value, 2))

    # Ensure the smallest non-zero value is not 0
    smallest_non_zero = min(
        value for value in standardized_values.values() if value != 0)
    if smallest_non_zero == 0:
        for category, value in standardized_values.items():
            if value != 0:
                # Set to a small non-zero value if all were 0
                standardized_values[category] = 1

    return standardized_values


def reorder_interests(df):
    # Define the desired order of interests
    order = ['conventional', 'investigative',
             'enterprising', 'artistic', 'social', 'realistic']

    # Set 'Interest' column as categorical with the desired order
    df['Interest'] = pd.Categorical(
        df['Interest'], categories=order, ordered=True)

    # Sort the DataFrame based on the 'Interest' column
    df_sorted = df.sort_values('Interest')

    # Reset the index
    df_sorted.reset_index(drop=True, inplace=True)

    return df_sorted


def interest_preprop(input):
    input.reset_index(inplace=True)
    input = input.rename(columns={'index': 'Interest'})
    interest_sorted = reorder_interests(input)

    # Preprop
    interest_pivot_transposed = interest_sorted.T
    interest_array = interest_pivot_transposed.values.reshape(1, -1)
    standardized_ranks_only = interest_array[:, 6:]
    input = standardized_ranks_only

    return input


def predicted_job(predictions, label_encoder, df_jobs_interest):

    # Get top 5 job indices
    top_5_indices = np.argsort(predictions[0])[-5:][::-1]
    # Decode the top 5 job indices to original job names using label_encoder
    top_5_classes = label_encoder.inverse_transform(top_5_indices)
    top_5_distances = predictions[0][top_5_indices]

    # Create a dataframe with job labels and their corresponding indices
    job_labels_with_indices = df_jobs_interest[['job_en']].reset_index()

    # Get the original job names using the indices
    top_5_jobs = job_labels_with_indices.loc[top_5_indices, 'job_en'].values

    predicted_jobs_and_distances = [
        (job, distance) for job, distance in zip(top_5_jobs, top_5_distances)
    ]
    return predicted_jobs_and_distances


def get_job_info(job_title):
    task_prompt = PromptTemplate.from_template(
        f"jelaskan dengan singkat task utama dari pekerjaan {job_title}!, sertakan contoh case singkat")
    task_chain = LLMChain(llm=llm, prompt=task_prompt)
    tasks = task_chain.run(prompt=task_prompt, max_tokens=50)

    work_style_prompt = PromptTemplate.from_template(
        f"jelaskan dengan singkat work styles dari pekerjaan {job_title}!")
    work_style_chain = LLMChain(llm=llm, prompt=work_style_prompt)
    work_styles = work_style_chain.run(prompt=work_style_prompt, max_tokens=50)

    return tasks.strip(), work_styles.strip()


def get_job_info_for_predictions(predicted_jobs_and_distances):
    tasks = []
    work_styles = []

    for job, _ in predicted_jobs_and_distances:
        task, work_style = get_job_info(job)
        tasks.append(task)
        work_styles.append(work_style)

    return tasks, work_styles


def output_for_job_recommend(input, task, work_style):
    df_predicted_jobs = pd.DataFrame(input, columns=['Job', 'Distance'])
    df_predicted_jobs['Tasks'] = task
    df_predicted_jobs['Work Styles'] = work_style
    return df_predicted_jobs


def combine_desc(top, bottom):
    df_talent = pd.read_csv("ml_sys/data/talents_desc(2).csv")

    # Top 10 talents
    df_talent_result_10 = pd.merge(
        top, df_talent, left_on='Talent', right_on='shorten_label')
    df_talent_result_10 = df_talent_result_10.drop(
        columns=['shorten_label', 'Unnamed: 0'])
    combined_text_top = " ".join(df_talent_result_10['positive_desc'])

    # Bottom 5 talents
    df_talent_result_5 = pd.merge(
        bottom, df_talent, left_on='Talent', right_on='shorten_label')
    df_talent_result_5 = df_talent_result_5.drop(
        columns=['shorten_label', 'Unnamed: 0'])
    combined_text_btm = " ".join(df_talent_result_5['negative_desc'])

    return combined_text_top, combined_text_btm


def get_summarization(top, bottom):
    task_prompt = PromptTemplate.from_template(
        f"Singkatkan deskripsi berikut! {top}. Buatkan penjelasan maksimal 5 kalimat dan minimal 4 kalimat")
    task_chain = LLMChain(llm=llm, prompt=task_prompt)
    summary_top = task_chain.run(prompt=task_prompt, max_tokens=400)

    task_prompt = PromptTemplate.from_template(
        f"Singkatkan deskripsi berikut! {bottom}. Buatkan penjelasan maksimal 3 kalimat dan minimal 2 kalimat")
    task_chain = LLMChain(llm=llm, prompt=task_prompt)
    summary_btm = task_chain.run(prompt=task_prompt, max_tokens=300)

    return summary_top.strip(), summary_btm.strip()


def output_talent(top, bottom):
    df_talent = pd.read_csv("ml_sys/data/talents_desc(2).csv")

    # Top 10 talents
    df_talent_result_10 = pd.merge(
        top, df_talent, left_on='Talent', right_on='shorten_label')
    df_talent_result_10 = df_talent_result_10[['name',
                                              'Predicted Rank', 'Strength']]

    # Bottom 5 talents
    df_talent_result_5 = pd.merge(
        bottom, df_talent, left_on='Talent', right_on='shorten_label')
    df_talent_result_5 = df_talent_result_5[['name',
                                            'Predicted Rank', 'Strength']]

    return df_talent_result_10, df_talent_result_5
