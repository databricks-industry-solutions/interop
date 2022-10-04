# Databricks notebook source
# MAGIC %md
# MAGIC ## 1. Data ingestion & interoperability: ingesting and processing FHIR bundles with dbignite
# MAGIC 
# MAGIC FHIR is a standard for health care data exchange, allowing entities to share data with interoperability.
# MAGIC 
# MAGIC Analysing and loading FHIR bundle can be hard, especially at scale.
# MAGIC 
# MAGIC As part of Databricks project, we developed [`dbignite`](https://github.com/databrickslabs/dbignite.git) to simplify FHIR ingestion. Using the library, you can accelerate your time to insight by:
# MAGIC 
# MAGIC * Parsing and reading the FHIR bundle out of the box
# MAGIC * Creating table for SQL usage (BI/reporting) on top of the incoming data
# MAGIC 
# MAGIC `dbignite` is available as a python wheel and can be installed as following:

# COMMAND ----------

# DBTITLE 1,Install dbignite using pip
# MAGIC %pip install https://hls-eng-data-public.s3.amazonaws.com/packages/dbignite-0.1.0-py2.py3-none-any.whl

# COMMAND ----------

# MAGIC %md
# MAGIC ### Analyzing raw FHIR bundles
# MAGIC Before we start, let's take a look at the files we want to ingest. They're generated data from synthea and available in our blob storage:

# COMMAND ----------

# DBTITLE 1,List FHIR bundles
BUNDLE_PATH="s3://hls-eng-data-public/data/synthea/fhir/fhir/"
files=dbutils.fs.ls(BUNDLE_PATH)
print(f'there are {len(files)} bundles to process')
display(files)

# COMMAND ----------

# DBTITLE 1,Take a look at one of the files
print(dbutils.fs.head(files[0].path))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### 1.1 Load FHIR bundles into ready to query Delta Tables
# MAGIC 
# MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/hls/resources/dbinterop/hls-dbiginte-flow-1.png" width="700px" style="float: right" />
# MAGIC 
# MAGIC We can leverage dbignite library to load this data using the `PersonDashboard` API. This will:
# MAGIC 
# MAGIC 1. Extract resources from FHIR bundles and create a dataframe where rows correspond to each patient bundle and columns contain extracted resources
# MAGIC 2. In addition, add corresponding tables - which have similar schemas to OMOP tabels - to our local database - (`print(dbName)`)

# COMMAND ----------

from dbignite.data_model import *

# COMMAND ----------

# DBTITLE 1,define database 
dbName='dbignite_demo' 
sql(f'DROP SCHEMA IF EXISTS {dbName} CASCADE;')

# COMMAND ----------

# DBTITLE 1,define fhir and cdm models
fhir_model=FhirBundles(BUNDLE_PATH)
cdm_model=OmopCdm(dbName)

# COMMAND ----------

# DBTITLE 1,define transformer
fhir2omop_transformer=FhirBundlesToCdm(spark)

# COMMAND ----------

# DBTITLE 1,transform from FHIR to CDM
fhir2omop_transformer.transform(fhir_model,cdm_model)

# COMMAND ----------

# DBTITLE 1,Transform from CDM to a patient dashboard 
cdm2dash_transformer=CdmToPersonDashboard(spark)
dash_model=PersonDashboard()
cdm2dash_transformer.transform(cdm_model,dash_model)
person_dashboard_df = dash_model.summary()

# COMMAND ----------

# DBTITLE 1,display all conditions for a given person
from pyspark.sql import functions as F
display(
   person_dashboard_df
  .filter("person_id='6efa4cd6-923d-2b91-272a-0f5c78a637b8'")
  .select(F.explode('conditions').alias('conditions'))
  .selectExpr('conditions.condition_start_datetime','conditions.condition_start_datetime','conditions.condition_status')
)

# COMMAND ----------

# DBTITLE 1,dbignite creates our tables out of the box:
# MAGIC %sql show tables

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### 1.2 Exploring the FHIR data
# MAGIC 
# MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/hls/resources/dbinterop/hls-dbiginte-flow-2.png" width="700px" style="float: right" />
# MAGIC 
# MAGIC Now that our data has been loaded, we can start running some exploration
# MAGIC 
# MAGIC #### Analyzing our patient informations
# MAGIC 
# MAGIC The FHIR bundle have been extracted and the patient information can now be analyzed using SQL or any python library. 
# MAGIC 
# MAGIC Let's see what's under the `person` table and start running some analysis on top of our dataset..

# COMMAND ----------

# MAGIC %sql select * from person

# COMMAND ----------

# DBTITLE 1,Analyzing Patient Age Distribution 
# MAGIC %sql
# MAGIC select gender_source_value, year(current_date())-year_of_birth as age, count(*) as count from person group by gender_source_value, age order by age
# MAGIC -- Plot: Key: age, group: gender_source_value, Values: count

# COMMAND ----------

# MAGIC %md
# MAGIC #### Patient Condition
# MAGIC 
# MAGIC The condition information are stored under the `condition` table

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from condition

# COMMAND ----------

# DBTITLE 1,Top condition per gender, using python and plotly
from pyspark.sql.functions import desc
import plotly.express as px

#We can also leverage pure SQL to access data
df = spark.table("person").join(spark.table("condition"), "person_id") \
          .groupBy(['gender_source_value', 'condition.condition_status']).count() \
          .orderBy(desc('count')).filter('count > 200').toPandas()
#And use our usual plot libraries
px.bar(df, x="condition_status", y="count", color="gender_source_value", barmode="group")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Procedure Occurrence Table
# MAGIC 
# MAGIC Let's explore the `procedure_occurrence` extracted from the bundles

# COMMAND ----------

# MAGIC %sql 
# MAGIC select * from procedure_occurrence

# COMMAND ----------

# DBTITLE 1,Distribution of procedures per-patient
df = spark.sql("select person_id, count(procedure_occurrence_id) as procedure_count from procedure_occurrence group by person_id having procedure_count < 500").toPandas()
px.histogram(df, x="procedure_count")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Encounter

# COMMAND ----------

# DBTITLE 1,Encounter repartition
df = spark.sql("select encounter_status, count(*) as count from encounter group by encounter_status order by count desc limit 20").toPandas()
px.pie(df, values='count', names='encounter_status', hole=.4)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## 2 Security and governance with Unity Catalog
# MAGIC 
# MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/hls/resources/dbinterop/hls-dbiginte-flow-3.png" width="700px" style="float: right; margin-left: 10px" />
# MAGIC 
# MAGIC Databricks Lakehouse add a layer of governance and security on all your resources.
# MAGIC 
# MAGIC We can choose to grant access to these resources to our Data Analysts so that they can have READ access only on these tables. This is done using standard SQL queries.
# MAGIC 
# MAGIC More advanced capabilities are available to support your sensitive use-cases:
# MAGIC 
# MAGIC * **Governance** and **traceability** with audit log (track who access what)
# MAGIC * **Fine grain access** / row level security (mask column containing PII information to a group of users)
# MAGIC * **Lineage** (track the downstream usage: who is using a given table and what are the impacts if you are to change it)

# COMMAND ----------

# MAGIC %md
# MAGIC as an example you can grant access to the following tables:
# MAGIC ```
# MAGIC GRANT SELECT ON TABLE person TO `my_analyst_user_group`;
# MAGIC GRANT SELECT ON TABLE procedure_occurrence TO `my_analyst_user_group`;
# MAGIC GRANT SELECT ON TABLE encounter_status TO `my_analyst_user_group`;
# MAGIC ```

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC 
# MAGIC ## 3 Building and sharing visualization (BI/DW)
# MAGIC 
# MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/hls/resources/dbinterop/hls-dbiginte-flow-4.png" width="700px"  style="float: right; margin-left: 10px" />
# MAGIC 
# MAGIC 
# MAGIC The lakehouse provides traditional Data warehousing and BI within one single platform and one security layer.
# MAGIC 
# MAGIC Now that our data is ready, we can leverage Databricks SQL capabilities to build interactive dashboard and share analysis. 
# MAGIC 
# MAGIC Open the [DBSQL Patient Summary](https://e2-demo-field-eng.cloud.databricks.com/sql/dashboards/352c784e-c0e3-4f42-8a66-a4955b7ee5f8-patient-summary?o=1444828305810485) as example.
# MAGIC 
# MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/hls/resources/dbinterop/hls-patient-dashboard.png" width="500"/>
