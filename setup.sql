-- Create or replace the database
CREATE OR REPLACE DATABASE InsuranceDB;

-- Create or replace the schema
CREATE OR REPLACE SCHEMA InsuranceDB.data;

-- Create or replace the warehouse
CREATE OR REPLACE WAREHOUSE InsuranceWarehouse
    WITH WAREHOUSE_SIZE = 'XSMALL'
    AUTO_SUSPEND = 300
    AUTO_RESUME = TRUE
    INITIALLY_SUSPENDED = TRUE;

-- Set the warehouse for use
USE WAREHOUSE InsuranceWarehouse;
-- Set the warehouse as default for the current user, if there is no default warehouse currently set
SET CUR_USER = (SELECT CURRENT_USER());
DESC USER IDENTIFIER($CUR_USER);
SET DEF_WAREHOUSE = (SELECT "value" FROM TABLE(RESULT_SCAN(LAST_QUERY_ID())) WHERE "property" = 'DEFAULT_WAREHOUSE');
SET WAREHOUSE_TO_SET = IFF(EQUAL_NULL($DEF_WAREHOUSE, 'null'), 'INSURANCEWAREHOUSE', $DEF_WAREHOUSE);
ALTER USER IDENTIFIER($CUR_USER) SET DEFAULT_WAREHOUSE = $WAREHOUSE_TO_SET;
    
-- Create or replace the Customers table
CREATE OR REPLACE TABLE InsuranceDB.data.Customers (
    client_id STRING PRIMARY KEY,
    name STRING,
    start_date DATE,
    total_claimed FLOAT,
    claims ARRAY, 
    state STRING,  
    city STRING,  
    zip_code STRING,  
    driver_license_state STRING
);

-- Create or replace the Claims table (now includes line_items)
CREATE OR REPLACE TABLE InsuranceDB.data.Claims (
    claim_number STRING PRIMARY KEY,
    client_id STRING,
    claim_type STRING,
    claim_amount FLOAT,
    status STRING,
    date_filed DATE,
    line_items ARRAY,  -- Stores structured line items
    FOREIGN KEY (client_id) REFERENCES InsuranceDB.data.Customers(client_id)
);

-- Create or replace an internal stage

CREATE STAGE CLAIM_STORAGE
    DIRECTORY = (ENABLE = TRUE)
    ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

CREATE OR REPLACE TABLE InsuranceDB.data.SupportDocs_Vectorized (
    file_name STRING,
    content STRING
);

CREATE OR REPLACE CORTEX SEARCH SERVICE support_docs_search
  ON content
  ATTRIBUTES file_name
  WAREHOUSE = insurancewarehouse
  TARGET_LAG = '1 hour'
  AS (
    SELECT
        file_name,
        content
    FROM InsuranceDB.data.SupportDocs_Vectorized
);

CREATE OR REPLACE STAGE insurance_data_stage
  URL = 's3://sfquickstarts/sfguide_getting_started_with_snowflake_agents_api_and_react/'
  FILE_FORMAT = (TYPE = 'CSV' FIELD_OPTIONALLY_ENCLOSED_BY = '"' SKIP_HEADER = 1);

COPY INTO InsuranceDB.data.Customers
FROM @insurance_data_stage/customers.csv
FILE_FORMAT = (TYPE = 'CSV' FIELD_OPTIONALLY_ENCLOSED_BY = '"' SKIP_HEADER = 1)
ON_ERROR = 'CONTINUE';

COPY INTO InsuranceDB.data.Claims
FROM @insurance_data_stage/claims.csv
FILE_FORMAT = (TYPE = 'CSV' FIELD_OPTIONALLY_ENCLOSED_BY = '"' SKIP_HEADER = 1)
ON_ERROR = 'CONTINUE';

COPY INTO InsuranceDB.data.SupportDocs_Vectorized
FROM @insurance_data_stage/supportdocs.csv
FILE_FORMAT = (TYPE = 'CSV' FIELD_OPTIONALLY_ENCLOSED_BY = '"' SKIP_HEADER = 1)
ON_ERROR = 'CONTINUE';

