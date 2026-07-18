import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat_ws, lit, lower, regexp_replace, trim
from pyspark.sql.types import StringType, StructField, StructType

# Configure Logging for production visibility
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MetadataProcessor")


def get_spark_session() -> SparkSession:
    """Initializes a local Spark session optimized for text/metadata processing."""
    logger.info("Initializing Spark Session...")
    return (
        SparkSession.builder.appName("GlobalHealthMetadataProcessor")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )


def define_unified_schema() -> StructType:
    """Defines the strict enterprise schema for the global health metadata repository."""
    return StructType(
        [
            StructField("source_dataset", StringType(), False),
            StructField("table_id", StringType(), True),
            StructField("variable_id", StringType(), False),
            StructField("variable_name", StringType(), True),
            StructField("description", StringType(), True),
        ]
    )


class MetadataTransformer:

    @staticmethod
    def transform_nhanes(df_raw) -> col:
        """Maps raw NHANES-like structures to the unified schema."""
        # Assuming raw NHANES has: Component, Item/Code, Description, Target
        return (
            df_raw.withColumn("source_dataset", lit("NHANES"))
            .withColumn("table_id", col("Component"))
            .withColumn("variable_id", col("Item/Code"))
            .withColumn("variable_name", col("Item/Code"))
            .withColumn("description", col("Description"))
            .select(
                "source_dataset",
                "table_id",
                "variable_id",
                "variable_name",
                "description",
            )
        )

    @staticmethod
    def transform_uk_biobank(df_raw) -> col:
        """Maps raw UK Biobank-like structures to the unified schema."""
        # Assuming raw UKB has: FieldID, Title, Description, Category
        return (
            df_raw.withColumn("source_dataset", lit("UK_Biobank"))
            .withColumn("table_id", concat_ws("_", lit("Cat"), col("Category")))
            .withColumn("variable_id", col("FieldID"))
            .withColumn("variable_name", col("Title"))
            .withColumn("description", col("Description"))
            .select(
                "source_dataset",
                "table_id",
                "variable_id",
                "variable_name",
                "description",
            )
        )


def build_composite_features(df):
    """Prepares text fields for downstream NLP, spaCy parsing, and Vector Embeddings."""
    logger.info("Enriching metadata with clean text composite features...")

    # Clean text: remove special characters, lowercase, remove trailing spaces
    clean_desc = regexp_replace(col("description"), r"[^a-zA-Z0-9\s\-_]", "")

    processed_df = (
        df.withColumn("description_clean", trim(lower(clean_desc)))
        .withColumn("variable_name_clean", trim(lower(col("variable_name"))))
        # Build a single robust string for the embedding model to read
        .withColumn(
            "composite_text",
            concat_ws(
                " | ",
                col("source_dataset"),
                col("variable_name_clean"),
                col("description_clean"),
            ),
        )
        .drop("description_clean", "variable_name_clean")
    )

    return processed_df


def main():
    spark = get_spark_session()

    # --- SIMULATING SOURCE DATA INGESTION ---
    # In a full run, these would be spark.read.csv() or spark.read.json()
    logger.info("Simulating source metadata loading...")

    nhanes_mock = spark.createDataFrame(
        [
            (
                "Demographics",
                "RIAGENDR",
                "Gender of the participant. 1=Male, 2=Female.",
            ),
            (
                "Examination",
                "BMXBMI",
                "Body Mass Index (kg/m**2) calculated for selected age ranges.",
            ),
        ],
        ["Component", "Item/Code", "Description"],
    )

    ukb_mock = spark.createDataFrame(
        [
            (
                "21001",
                "Body mass index (BMI)",
                "Anthropometric measurement of weight divided by squared height.",
                "1001",
            ),
            (
                "31",
                "Sex",
                "Genetic or phenotypic sex reported by participant.",
                "1002",
            ),
        ],
        ["FieldID", "Title", "Description", "Category"],
    )

    # --- TRANSFORMATION & STANDARDIZATION ---
    logger.info("Standardizing multi-source schemas into unified dataframes...")
    unified_nhanes = MetadataTransformer.transform_nhanes(nhanes_mock)
    unified_ukb = MetadataTransformer.transform_uk_biobank(ukb_mock)

    # Combine all international datasets via Union
    global_atlas_df = unified_nhanes.union(unified_ukb)

    # --- NLP TEXT PREPARATION ---
    final_atlas_df = build_composite_features(global_atlas_df)

    # --- WRITE OUT TO DISK (Parquet Format for efficient retrieval) ---
    output_path = "data/processed_metadata_atlas"
    logger.info(f"Saving fully unified metadata warehouse to {output_path}...")

    # Save as parquet - perfect for quick reads when converting to vector spaces
    final_atlas_df.write.mode("overwrite").parquet(output_path)

    # Show showcase output snippet in logs
    final_atlas_df.show(truncate=False)
    logger.info("Step 1 Pipeline Completed Successfully.")


if __name__ == "__main__":
    main()
    