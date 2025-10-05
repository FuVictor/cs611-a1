import os
from pathlib import Path
from tkinter.font import names
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

BRONZE_DIR = Path("datamart/bronze")
SILVER_DIR = Path("datamart/silver")
GOLD_DIR = Path("datamart/gold")
DATA_DIR = Path("data")

def ensure_dirs():
    for p in [BRONZE_DIR, SILVER_DIR, GOLD_DIR]:
        p.mkdir(parents=True, exist_ok=True)

def build_spark():
    return (
        SparkSession.builder
        .appName("CS611-A1-Medallion")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .getOrCreate()
    )

# Bronze: load raw CSV
def bronze_ingest_csv(spark):
    
    def read_csv_try(names):
        for n in names:
            p = DATA_DIR / n
            if p.exists():
                df = (
                    spark.read.option("header", True)
                    .option("inferSchema", True)
                    .csv(str(p))
                )
                return df.withColumn("_ingest_ts", F.current_timestamp())
        raise FileNotFoundError(f"None of these files found in data/: {names}")
    

    click = read_csv_try(["feature_clickstream.csv", "_feature_clickstream.csv"])
    attr  = read_csv_try(["feature_attributes.csv", "features_attributes.csv", "_feature_attributes.csv", "_features_attributes.csv"])
    fin   = read_csv_try(["feature_financials.csv", "features_financials.csv", "_feature_financials.csv", "_features_financials.csv"])
    loan  = read_csv_try(["lms_loan_daily.csv", "_lms_loan_daily.csv"])


    click.write.mode("overwrite").parquet(str(BRONZE_DIR / "feature_clickstream"))
    attr.write.mode("overwrite").parquet(str(BRONZE_DIR / "feature_attributes"))
    fin.write.mode("overwrite").parquet(str(BRONZE_DIR / "feature_financials"))
    loan.write.mode("overwrite").parquet(str(BRONZE_DIR / "lms_loan_daily"))

# Silver: cleaning
def silver_clean_and_standardize(spark):
    def load(bronze_name):
        return spark.read.parquet(str(BRONZE_DIR / bronze_name))

    def to_snake(df):
        for c in df.columns:
            df = df.withColumnRenamed(c, c.strip().lower().replace(" ", "_"))
        return df

    click = to_snake(load("feature_clickstream")).dropDuplicates()
    attr  = to_snake(load("feature_attributes")).dropDuplicates()
    fin   = to_snake(load("feature_financials")).dropDuplicates()
    loan  = to_snake(load("lms_loan_daily")).dropDuplicates()

    click.write.mode("overwrite").parquet(str(SILVER_DIR / "feature_clickstream"))
    attr.write.mode("overwrite").parquet(str(SILVER_DIR / "feature_attributes"))
    fin.write.mode("overwrite").parquet(str(SILVER_DIR / "feature_financials"))
    loan.write.mode("overwrite").parquet(str(SILVER_DIR / "lms_loan_daily"))

# Gold: simple feature store  (FIXED)
def gold_build_feature_store(spark):
    click = spark.read.parquet(str(SILVER_DIR / "feature_clickstream"))
    attr  = spark.read.parquet(str(SILVER_DIR / "feature_attributes"))
    fin   = spark.read.parquet(str(SILVER_DIR / "feature_financials"))

  
    def unify_user_id(df):
        for cand in ["user_id", "customer_id", "account_id"]:
            if cand in df.columns:
                return df if cand == "user_id" else df.withColumnRenamed(cand, "user_id")
        raise ValueError("Missing user_id/customer_id/account_id column")

    click = unify_user_id(click)
    attr  = unify_user_id(attr)
    fin   = unify_user_id(fin)


    click_agg = click.groupBy("user_id").agg(F.count(F.lit(1)).alias("clicks_all"))

 
    attr_agg = attr.dropDuplicates(["user_id"])

 
    def _is_numeric(dtype: str) -> bool:
        d = dtype.lower()
        return (
            d in ("int", "bigint", "double", "float", "long", "short", "byte")
            or d.startswith("decimal")
        )

    fin_num_cols = [c for c, t in fin.dtypes if c != "user_id" and _is_numeric(t)]
    if fin_num_cols:
        fin_agg = fin.groupBy("user_id").agg(*[F.avg(c).alias(f"avg_{c}") for c in fin_num_cols])
    else:
       
        fin_agg = fin.select("user_id").dropDuplicates()

  
    feat = (
        click_agg
        .join(attr_agg, "user_id", "left")
        .join(fin_agg,  "user_id", "left")
        .withColumn("feature_snapshot_ts", F.current_timestamp())
    )
    feat.write.mode("overwrite").parquet(str(GOLD_DIR / "features_user"))


def main():
    ensure_dirs()
    spark = build_spark()
    try:
        bronze_ingest_csv(spark)
        silver_clean_and_standardize(spark)
        gold_build_feature_store(spark)
        print("✅ Pipeline finished, check datamart/")
    finally:
        spark.stop()

if __name__ == "__main__":
    main()
