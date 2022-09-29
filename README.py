# Databricks notebook source
# MAGIC %md
# MAGIC # Accelerating Interoperability With Databricks Lakehouse 
# MAGIC ## From FHIR ingestion to patient outcomes analysis
# MAGIC <br/>
# MAGIC <br/>
# MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/hls/resources/dbinterop/hls-dbiginte-lakehouse.png" width="1000px" />
# MAGIC 
# MAGIC In this solution accelerator, we demonstrate how we can leverage the lakehouse approach, for an in-depth analysis of patient outcomes,
# MAGIC using EHR data.
# MAGIC Consider a scenario that we have a collection of [FHIR](https://www.hl7.org/fhir/overview.html) bundles and want to explore the effect of different factors on Covid outcomes. However, FHIR standard is primarily designed for the exchange of information and not optimized for analytics. To solve this problem, we need to flatten 
# MAGIC the the bundles (stored as nested `json` files) and extract resources such as patients, encounters, conditions etc. so that we can create a dataset which is 
# MAGIC ready for exploratory data analysis.
# MAGIC We can decompose this process in 3 main steps:
# MAGIC * **Data ingestion** (on the left)
# MAGIC   * Simplify ingestion, from all kind of sources. As example, we'll use _[Databricks Labs `dbignite` library](https://github.com/databrickslabs/dbignite.git)_ to ingest FHIR bundle as tables ready to be queried in SQL in one line.
# MAGIC   * Query and explore the data ingested
# MAGIC   * Optionaly we can secure data access
# MAGIC   
# MAGIC * **Eploratory Analysis/Data Curation** (flow on the top)
# MAGIC   * Create cohorts
# MAGIC   * Create a patient level data strucure (a patient dashboard) from the bundles
# MAGIC   * Investigate rate of hospital admissions among covid patients and explore correlations among different factors such as SDOH, disease history and hospital admission
# MAGIC   
# MAGIC * **Data Science / Advance Analystics** (bottom)
# MAGIC   * Create patient features 
# MAGIC   * Create a training dataset to build a model predicting and analysing our cohort 
# MAGIC   * Use SHAP for explaining the effect of different features on the outcome under study
# MAGIC   
# MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/hls/resources/dbinterop/hls-dbiginte-flow-0.png" width="1000px"/>
# MAGIC 
# MAGIC ### Data
# MAGIC The data used in this demo is generated using [synthea](https://synthetichealth.github.io/synthea/). We used [covid infections module](https://github.com/synthetichealth/synthea/blob/master/src/main/resources/modules/covid19/infection.json), which incorporates patient risk factors such as diabetes, hypertension and SDOH in determining outcomes. The data is available at `s3://hls-eng-data-public/data/synthea/fhir/fhir/`. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## License ⚖️
# MAGIC Copyright / License info of the notebook. Copyright Databricks, Inc. [2022].  The source in this notebook is provided subject to the [Databricks License](https://databricks.com/db-license-source).  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC |Library Name|Library License|Library License URL|Library Source URL| 
# MAGIC | :-: | :-:| :-: | :-:|
# MAGIC |Synthea|Apache License 2.0|https://github.com/synthetichealth/synthea/blob/master/LICENSE| https://github.com/synthetichealth/synthea|
# MAGIC |The Book of OHDSI | Creative Commons Zero v1.0 Universal license.|https://ohdsi.github.io/TheBookOfOhdsi/index.html#license|https://ohdsi.github.io/TheBookOfOhdsi/|
# MAGIC 
# MAGIC ### Disclaimers
# MAGIC *Databricks Inc. (“Databricks”) does not dispense medical, diagnosis, or treatment advice. This demo (“tool”) is for informational purposes only and may not be used as a substitute for professional medical advice, treatment, or diagnosis. This tool may not be used within Databricks to process Protected Health Information (“PHI”) as defined in the Health Insurance Portability and Accountability Act of 1996, unless you have executed with Databricks a contract that allows for processing PHI, an accompanying Business Associate Agreement (BAA), and are running this notebook within a HIPAA Account.  Please note that if you run this notebook within Azure Databricks, your contract with Microsoft applies.*
