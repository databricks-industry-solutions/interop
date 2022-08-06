# Analysis of FHIR Bundles using SQL and Python

<img src="http://hl7.org/fhir/assets/images/fhir-logo-www.png" width = 10%>

In this workshop: 
  1. We use datarbicks `dbinterop` package to ingest FHIR bundles (in `json` format) into deltalake
  2. Create a patient-level dashboard from the bundles
  3. Create cohorts
  4. Investigate rate of hospital admissions among covid patients and explore the effect of SDOH and disease history in hospital admissions
  5. Create a feature store of patient features, and use the feature store to create a training dataset for downstream ML workloads, using [databricks feature store](https://docs.databricks.com/applications/machine-learning/feature-store/index.html#databricks-feature-store). 
<br>
</br>
<img src="https://hls-eng-data-public.s3.amazonaws.com/img/FHIR-RA.png" width = 50%>

### Data
The data used in this demo is generated using [synthea](https://synthetichealth.github.io/synthea/). We used [covid infections module](https://github.com/synthetichealth/synthea/blob/master/src/main/resources/modules/covid19/infection.json), which incorporates patient risk factors such as diabetes, hypertension and SDOH in determining outcomes. The data is available at `s3://hls-eng-data-public/data/synthea/fhir/fhir/`. 
___

### License
Copyright / License info of the notebook. Copyright Databricks, Inc. [2021].  The source in this notebook is provided subject to the [Databricks License](https://databricks.com/db-license-source).  All included or referenced third party libraries are subject to the licenses set forth below.

|Library Name|Library License|Library License URL|Library Source URL| 
| :-: | :-:| :-: | :-:|
|Synthea|Apache License 2.0|https://github.com/synthetichealth/synthea/blob/master/LICENSE| https://github.com/synthetichealth/synthea|
|The Book of OHDSI | Creative Commons Zero v1.0 Universal license.|https://ohdsi.github.io/TheBookOfOhdsi/index.html#license|https://ohdsi.github.io/TheBookOfOhdsi/|

### Disclaimers
Databricks Inc. (“Databricks”) does not dispense medical, diagnosis, or treatment advice. This Solution Accelerator (“tool”) is for informational purposes only and may not be used as a substitute for professional medical advice, treatment, or diagnosis. This tool may not be used within Databricks to process Protected Health Information (“PHI”) as defined in the Health Insurance Portability and Accountability Act of 1996, unless you have executed with Databricks a contract that allows for processing PHI, an accompanying Business Associate Agreement (BAA), and are running this notebook within a HIPAA Account.  Please note that if you run this notebook within Azure Databricks, your contract with Microsoft applies.
