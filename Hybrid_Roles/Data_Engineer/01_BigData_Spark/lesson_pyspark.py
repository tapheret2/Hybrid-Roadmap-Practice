"""
================================================================
DATA ENGINEER - MODULE 1: BIG DATA WITH PYSPARK
================================================================

Data Engineer xây dựng infrastructure để xử lý dữ liệu lớn
Spark là công cụ phổ biến nhất cho big data processing

Cài đặt: pip install pyspark
"""

# --- 1. LÝ THUYẾT (THEORY) ---
"""
1. Tại sao cần Big Data Tools?
   - Dữ liệu quá lớn để fit vào RAM của 1 máy
   - Cần parallelism để xử lý nhanh
   - Pandas: GB scale, Spark: TB/PB scale

2. Spark Core Concepts:
   - Driver: Điều phối, lập kế hoạch
   - Executors: Workers thực thi tasks
   - RDD: Resilient Distributed Dataset (low-level)
   - DataFrame: Higher-level API (recommended)

3. Lazy Evaluation:
   - Transformations (map, filter) không chạy ngay
   - Chỉ chạy khi gọi Actions (collect, show, write)
   - Cho phép optimization

4. Partitioning:
   - Dữ liệu được chia thành partitions
   - Mỗi partition xử lý độc lập
   - Quan trọng cho performance
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType

# --- 2. CODE MẪU (CODE SAMPLE) ---

def create_spark_session():
    """Khởi tạo Spark session"""
    spark = SparkSession.builder \
        .appName("DataEngineerLearning") \
        .master("local[*]") \
        .config("spark.sql.shuffle.partitions", "4") \
        .getOrCreate()
    return spark

def basic_operations(spark):
    """Các operations cơ bản với Spark DataFrame"""
    
    # Tạo DataFrame từ data
    data = [
        ("Alice", "IT", 5000, 2020),
        ("Bob", "HR", 4500, 2019),
        ("Charlie", "IT", 6000, 2018),
        ("Diana", "Finance", 7000, 2020),
        ("Eve", "IT", 5500, 2021),
        ("Frank", "HR", 4800, 2022),
    ]
    
    columns = ["name", "department", "salary", "join_year"]
    df = spark.createDataFrame(data, columns)
    
    # Show data
    print("=== Original Data ===")
    df.show()
    df.printSchema()
    
    # SELECT & WHERE
    print("=== IT Department ===")
    df.filter(F.col("department") == "IT") \
      .select("name", "salary") \
      .show()
    
    # Multiple conditions
    print("=== High Salary Recent Hires ===")
    df.filter((F.col("salary") > 5000) & (F.col("join_year") >= 2020)) \
      .show()
    
    # Add/Modify columns
    print("=== With Bonus ===")
    df.withColumn("bonus", F.col("salary") * 0.1) \
      .withColumn("total", F.col("salary") + F.col("salary") * 0.1) \
      .show()
    
    return df

def aggregations(spark, df):
    """Aggregations và GroupBy"""
    
    print("=== Aggregations ===")
    
    # Basic agg
    df.agg(
        F.count("*").alias("total_employees"),
        F.avg("salary").alias("avg_salary"),
        F.max("salary").alias("max_salary"),
        F.min("salary").alias("min_salary")
    ).show()
    
    # GroupBy
    print("=== Group By Department ===")
    df.groupBy("department") \
      .agg(
          F.count("*").alias("count"),
          F.avg("salary").alias("avg_salary"),
          F.sum("salary").alias("total_salary")
      ) \
      .orderBy(F.desc("avg_salary")) \
      .show()

def window_functions(spark, df):
    """Window Functions - Rất quan trọng cho Data Engineering"""
    
    print("=== Window Functions ===")
    
    # Define window
    dept_window = Window.partitionBy("department").orderBy(F.desc("salary"))
    
    # Rank within department
    df.withColumn("rank", F.rank().over(dept_window)) \
      .withColumn("dense_rank", F.dense_rank().over(dept_window)) \
      .show()
    
    # Running total
    year_window = Window.orderBy("join_year").rowsBetween(Window.unboundedPreceding, 0)
    
    print("=== Running Total by Year ===")
    df.withColumn("running_employees", F.count("*").over(year_window)) \
      .withColumn("running_salary", F.sum("salary").over(year_window)) \
      .orderBy("join_year") \
      .show()

def joins_demo(spark):
    """Joins trong Spark"""
    
    employees = spark.createDataFrame([
        (1, "Alice", 101),
        (2, "Bob", 102),
        (3, "Charlie", 101),
        (4, "Diana", 103),
    ], ["emp_id", "name", "dept_id"])
    
    departments = spark.createDataFrame([
        (101, "Engineering", "Building A"),
        (102, "Marketing", "Building B"),
        (104, "Finance", "Building C"),  # No employees
    ], ["dept_id", "dept_name", "location"])
    
    print("=== Inner Join ===")
    employees.join(departments, "dept_id", "inner").show()
    
    print("=== Left Join (all employees) ===")
    employees.join(departments, "dept_id", "left").show()
    
    print("=== Right Join (all departments) ===")
    employees.join(departments, "dept_id", "right").show()

def read_write_data(spark):
    """Đọc/Ghi dữ liệu với các formats"""
    
    # Read CSV
    # df = spark.read.csv("data.csv", header=True, inferSchema=True)
    
    # Read Parquet (recommended for big data)
    # df = spark.read.parquet("data.parquet")
    
    # Read JSON
    # df = spark.read.json("data.json")
    
    # Write Parquet (partitioned)
    # df.write.partitionBy("year", "month") \
    #   .mode("overwrite") \
    #   .parquet("output/")
    
    # Write to Delta Lake (ACID transactions)
    # df.write.format("delta").save("delta_table/")
    
    print("=== Read/Write Demo ===")
    print("Xem code comments để biết cách đọc/ghi các formats")

# --- 3. BÀI TẬP (EXERCISE) ---
"""
BÀI 1: Tạo DataFrame với 1 triệu records (dùng range và random)
       - Tính thời gian xử lý với các transformations
       - So sánh với Pandas

BÀI 2: Implement ETL pipeline:
       - Extract: Đọc CSV files
       - Transform: Clean, aggregate, join
       - Load: Ghi ra Parquet partitioned by date

BÀI 3: Optimize performance:
       - Repartition data hợp lý
       - Cache intermediate results
       - Broadcast join cho small tables

BÀI 4: Streaming với Spark Structured Streaming:
       - Đọc từ socket hoặc Kafka
       - Window aggregations
       - Output to console/file
"""

# --- TEST ---
if __name__ == "__main__":
    print("=== PySpark Big Data Processing ===\n")
    
    spark = create_spark_session()
    
    df = basic_operations(spark)
    aggregations(spark, df)
    window_functions(spark, df)
    joins_demo(spark)
    read_write_data(spark)
    
    spark.stop()
