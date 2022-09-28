# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # Advanced Analytics: Covid Outcomes Analysis
# MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/hls/resources/dbinterop/hls-dbiginte-flow-5.png" width="700px"  style="float: right; margin-left: 10px" />
# MAGIC 
# MAGIC In this notebook, we demonstrate how databricks lakehouse platform can be used for exploratory data science and explainable AI.
# MAGIC In this example we focus on Covid outcomes and explore different factors, such as SDOH and disease history and thier impact on outcomes.
# MAGIC 
# MAGIC To this end, we first create cohorts corresponding to features under consideration and then explore correlations among different features as well as correlation between each feature and the outcome under study.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### 1. Cohort definition
# MAGIC To ensure better reproducibility and data organization, we first create patient cohorts based on the criteria of interest (being admitted to hospital, infection status, disease history etc). We then proceed to create features based on cohorts and add the results to databricks feature store.
# MAGIC 
# MAGIC To make data access easier we add cohort tables to our database (similar to OMOP's results schema). See [The Book Of OHDSI](https://ohdsi.github.io/TheBookOfOhdsi/CommonDataModel.html#cdm-standardized-tables) for more detail.

# COMMAND ----------

# DBTITLE 1,set up cohort schema
COHORT_SCHEMA_NAME = 'dbignite_demo_cohorts'
EHR_SCHEMA_NAME = 'dbignite_demo'

sql(f'DROP SCHEMA IF EXISTS {COHORT_SCHEMA_NAME} CASCADE;')
sql(f'create schema {COHORT_SCHEMA_NAME}')
sql(f'use {EHR_SCHEMA_NAME}')

# COMMAND ----------

# DBTITLE 1,Create tables
sql(f"""
CREATE TABLE IF NOT EXISTS {COHORT_SCHEMA_NAME}.cohort (
  cohort_definition_id INT,
  person_id STRING,
  cohort_start_date DATE,
  cohort_end_date DATE
  )
  """
)

sql(f"""CREATE TABLE IF NOT EXISTS {COHORT_SCHEMA_NAME}.cohort_definition (
  cohort_definition_id INT,
  cohort_definition_name STRING,
  cohort_definition_description STRING,
  cohort_definition_syntax STRING,
  cohort_initiation_date DATE
  )
  """
)
#to reset existing cohort if any
#sql(f"DELETE from {COHORT_SCHEMA_NAME}.cohort") 
#sql(f"DELETE from {COHORT_SCHEMA_NAME}.cohort_definition")

# COMMAND ----------

sql(f'describe schema {COHORT_SCHEMA_NAME}').display()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Creating the cohorts
# MAGIC 
# MAGIC We can now run simple SQL queries create the cohort and fill the `cohort` and `cohort_definition`. To make cohort creation simpler (and re-usable) we can define functions that create cohorts, given a set of parameters such as inclusion/exclusion criteria. 

# COMMAND ----------

# DBTITLE 1,Define cohort creation function
from pyspark.sql.functions import lit, array_contains, year, first, collect_list
import uuid
def create_cohort(cohort_definition_name, cohort_definition_description, cohort):
  uid=abs(uuid.uuid4().fields[0])
  print(f"saving cohort {cohort_definition_name} under the cohort table (id {uid}). Adding entry to cohort_definition.")
  #Add the cohort to the cohort table
  spark.sql(cohort).withColumn("cohort_definition_id", lit(uid).cast('int')).write.mode("append").saveAsTable(f"{COHORT_SCHEMA_NAME}.cohort")
  #Add the cohort definition to the metadata table
  sql(f"""
      INSERT INTO {COHORT_SCHEMA_NAME}.cohort_definition
        select int({uid}) as cohort_definition_id,
        '{cohort_definition_name}' as cohort_definition_name,
        '{cohort_definition_description}' as cohort_definition_description,
        '{cohort}' as cohort_definition_syntax,
        current_date() as cohort_initiation_date""")

# COMMAND ----------

# DBTITLE 0,create and store cohorts
patient_cohort = """select person_id, to_date(condition_start_datetime) as cohort_start_date, to_date(condition_end_datetime) as cohort_end_date 
                           from condition where condition_code in (840539006)"""

admission_cohort = """select person_id, to_date(first(encounter_period_start)) as cohort_start_date, to_date(first(encounter_period_end)) as cohort_end_date 
                                from encounter where encounter_code in (1505002, 32485007, 305351004, 76464004) group by person_id"""

#You can open the create_cohort function in the companion notebook in ./_resources
create_cohort('covid', 'patient with covid', patient_cohort)
create_cohort('admission', 'patients admitted', admission_cohort)

# COMMAND ----------

sql(f"use {COHORT_SCHEMA_NAME}")

# COMMAND ----------

# DBTITLE 1,Our 2 cohorts are created: 'covid' and 'admission'
# MAGIC %sql
# MAGIC select * from cohort_definition

# COMMAND ----------

# DBTITLE 1,Reviewing the admission cohort
# MAGIC %sql
# MAGIC select c.cohort_definition_id, person_id, cohort_start_date, cohort_end_date from cohort c join cohort_definition cd
# MAGIC   where cd.cohort_definition_id = c.cohort_definition_id and cd.cohort_definition_name = 'admission';

# COMMAND ----------

# DBTITLE 1,Percentage of covid patients admitted to the hospital
covid_admissions_df = spark.sql("""
  with 
    covid as (select c.person_id, cohort_start_date as covid_start from cohort c join cohort_definition cd using (cohort_definition_id) where cd.cohort_definition_name='covid'),
    admission as (select person_id, first(cohort_start_date) as admission_start from cohort c join cohort_definition cd using (cohort_definition_id) where cd.cohort_definition_name='admission' group by person_id)
  select
    case when admission_start between covid_start AND covid_start + interval 30 days then 1 else 0 end as is_admitted, * from covid left join admission using(person_id)""")

covid_admissions_df.selectExpr('100*avg(is_admitted) as percent_admitted').display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Analyze correlation between different factors in our cohorts
# MAGIC 
# MAGIC Now let's take a deeper look into the correlations between different factors. 
# MAGIC 
# MAGIC To do that, we'll add disease history and SDOH information to our patient cohort
# MAGIC 
# MAGIC To simplify downstream analysis, we'll flatten the patient conditions and add 1 column per condition (`True` or `False`)
# MAGIC 
# MAGIC *Note: here, we directly create a dataset of disease and SDOH histories, represented as binary values. Alternatively, for each condition and a given timeframe, you can add a cohort of patients, having had that condition and add to the cohort table.*

# COMMAND ----------

# DBTITLE 1,patient comorbidity history
#Condition code we're interested in
conditions_list = {
  "full-time-employment": 160903007,
  "part-time-employment": 160904001,
  "not-in-labor-force": 741062008,
  "received-higher-education": 224299000,
  "has-a-criminal-record": 266948004,
  "unemployed": 73438004,
  "refugee": 446654005,
  "misuses-drugs": 361055000,
  "obesity": 162864005,
  "prediabetes": 15777000,
  "hypertension": 59621000,
  "diabetes": 44054006,
  "coronary-heart-disease": 53741008
}


#Add 1 column per condition (true or false)
def create_patient_history_table(conditions_list):
  patient_history_df = spark.sql(f'select person_id, first(gender_source_value) as gender, year(current_date) - first(year_of_birth) as age, collect_list({EHR_SCHEMA_NAME}.condition.condition_code) as conditions from {EHR_SCHEMA_NAME}.person left join {EHR_SCHEMA_NAME}.condition using (person_id) group by person_id')
  for cond_name, cond_code in conditions_list.items():
    patient_history_df = patient_history_df.withColumn(f'history_of_{cond_name}', array_contains('conditions', str(cond_code)))
    
  return patient_history_df.drop('conditions')

patient_history_df = create_patient_history_table(conditions_list)
display(patient_history_df)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC To conduct a complete analysis we look at the [mutual information](https://en.wikipedia.org/wiki/Mutual_information) between different features in our dataset. 
# MAGIC 
# MAGIC To calculate mutual information we use [`normalized_mutual_info_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.normalized_mutual_info_score.html) from `sklearn`.

# COMMAND ----------

# DBTITLE 1,define function to calculate MI
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics.cluster import normalized_mutual_info_score

def get_normalized_mutual_info_score(df):
  cols=patient_covid_hist_pdf.columns
  
  ll=[]
  for col1 in cols:
    for col2 in cols:
      ll+=[normalized_mutual_info_score(patient_covid_hist_pdf[col1], patient_covid_hist_pdf[col2])]
  cols = [m.replace('history_of_','') for m in cols]
  mutI_pdf = pd.DataFrame(np.array(ll).reshape(len(cols),len(cols)),index=cols,columns=cols)
  return mutI_pdf**(1/3)

# COMMAND ----------

#join patient history with covid admission history 
patient_covid_hist = patient_history_df.join(covid_admissions_df.select('person_id', 'is_admitted'), on='person_id')
#save it as table for future analysis
patient_covid_hist.write.mode('overwrite').saveAsTable("patient_covid_hist")
patient_covid_hist_pdf = spark.table("patient_covid_hist").drop('person_id', 'gender', 'age').toPandas()
#For details see implementation in the companion notebook using sklearn normalized_mutual_info_score
plot_pdf = get_normalized_mutual_info_score(patient_covid_hist_pdf)
plot_pdf.style.background_gradient(cmap='Blues').format(precision=2)

# COMMAND ----------

# MAGIC %md
# MAGIC Looking at the table above, we see that the highest correlation is between **hospital admissions** and **hypertension**, followed by **coronary-heart-disease**, which seems consistent with the factors taken into account in the synthea module for covid infections. 
# MAGIC 
# MAGIC On the SDOH side, we see high correlations with part-time employment status. However, we also see high correlation with criminal records which can be example of a spurious correlation due to small sample size (100).

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC 
# MAGIC ## 3. Advanced analytics: Predicting the risk of being admitted with COVID
# MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/hls/resources/dbinterop/hls-dbiginte-flow-6.png" width="700px"  style="float: right; margin-left: 10px" />
# MAGIC 
# MAGIC As a next step, we'll train binary classifier to predict the outcome (`is_admitted`) based on the features provided in the training data.
# MAGIC 
# MAGIC This model will predict the likelyhood of behing admitted and will help us understanding which feature has the most impact for the model globally but also each patient.
# MAGIC 
# MAGIC We'll use our previous cohort to run this analysis:

# COMMAND ----------

cohort_training = patient_covid_hist.toPandas()
display(cohort_training)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### 3.1. Training our model, leveraging MLFlow and AutoML
# MAGIC 
# MAGIC We'll train a simple `XGBClassifier` on our model, trying to predict the `is_admitted`.
# MAGIC 
# MAGIC We'll leverage Databricks ML capabilities, including MLFlow to track all our experimentation out of the box, providing among other traceability and reproducibility.
# MAGIC 
# MAGIC Note that to accelerate the model creation, we could also have use [Databricks Auto-ML](https://www.databricks.com/product/automl), producing a state of the art notebook with a model ready to be used. <br/>
# MAGIC This typically saves days while providing best practices to the team.

# COMMAND ----------

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
#Cleanup and Split data before training the model: 
# - remove 'history_of' for more visibility in the column name
# - change gender as boolean column for simplicity
# - split in training/test dataset
def prep_data_for_classifier(training):
  X = training.drop(labels=['person_id', 'is_admitted'], axis=1)
  to_rename = {'gender': 'is_male'}
  for c in X.columns:
    #Drop the "history_of_"
    if "history_of" in c:
      to_rename[c] = c[len("history_of_"):]
  X['gender'] = X['gender'] == 'male'
  X = X.rename(columns=to_rename)
  Y = training['is_admitted']
  return train_test_split(X, Y)

# COMMAND ----------

# Start an mlflow run, which is needed for the feature store to log the model
with mlflow.start_run() as run: 
  # Build a small model for this example
  X_train, X_test, y_train, y_test = prep_data_for_classifier(cohort_training)
  model = Pipeline([('classifier', XGBClassifier())])
  model.fit(X_train, y_train)
  #Log the model & deploy it to our registry for further usage. This will also link it to our notebook.
  mlflow.sklearn.log_model(model, "model",  registered_model_name="field_demos_covid_prediction")
  print(f"Model trained. Accuracy score: {model.score(X_test, y_test)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2 Model Analysis
# MAGIC 
# MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/hls/resources/dbinterop/hls-dbiginte-flow-7.png" width="700px"  style="float: right; margin-left: 10px" />
# MAGIC 
# MAGIC Our model is now trained. Open the right Experiment menu to access MLFlow UI and the model saved in the registry.
# MAGIC 
# MAGIC Using this model, we can predict the probability of a patient being admitted and analyze which features are important to determine that.
# MAGIC 
# MAGIC We'll explain the feature importance in our model and visualize the feature impact on the predictions.

# COMMAND ----------

import shap
explainer = shap.TreeExplainer(model['classifier'])
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values = shap_values, features = X_train)

# COMMAND ----------

# MAGIC %md
# MAGIC As expected, the **Hypertension** and **Age** of our patient have the strongest impact in our model.
# MAGIC 
# MAGIC Let's predict the probability of admission of a patient and understand which features drive this outcome:

# COMMAND ----------

# MAGIC %md
# MAGIC ## License ⚖️
# MAGIC Copyright / License info of the notebook. Copyright Databricks, Inc. [2022].  The source in this notebook is provided subject to the [Databricks License](https://databricks.com/db-license-source).  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC 
# MAGIC |Library Name|Library License|Library License URL|Library Source URL| 
# MAGIC | :-: | :-:| :-: | :-:|
# MAGIC |Synthea|Apache License 2.0|https://github.com/synthetichealth/synthea/blob/master/LICENSE| https://github.com/synthetichealth/synthea|
# MAGIC |The Book of OHDSI | Creative Commons Zero v1.0 Universal license.|https://ohdsi.github.io/TheBookOfOhdsi/index.html#license|https://ohdsi.github.io/TheBookOfOhdsi/|
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### Disclaimers
# MAGIC *Databricks Inc. (“Databricks”) does not dispense medical, diagnosis, or treatment advice. This demo (“tool”) is for informational purposes only and may not be used as a substitute for professional medical advice, treatment, or diagnosis. This tool may not be used within Databricks to process Protected Health Information (“PHI”) as defined in the Health Insurance Portability and Accountability Act of 1996, unless you have executed with Databricks a contract that allows for processing PHI, an accompanying Business Associate Agreement (BAA), and are running this notebook within a HIPAA Account.  Please note that if you run this notebook within Azure Databricks, your contract with Microsoft applies.*
